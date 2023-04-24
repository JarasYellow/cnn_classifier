from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

def make_train_step(model, loss_fnc, optimizer):
    def train_step(X,Y):
        # set model to train mode
        model.train()
        # forward pass
        output_logits, output_softmax, attention_weights_norm = model(X)
        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y==predictions)/float(len(Y))
        # compute loss
        loss = loss_fnc(output_logits, Y)
        # compute gradients
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy*100
    return train_step

def make_validate_fnc(model,loss_fnc):
    def validate(X,Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax, attention_weights_norm = model(X)
            predictions = torch.argmax(output_softmax,dim=1)
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            loss = loss_fnc(output_logits,Y)
        return loss.item(), accuracy*100, predictions
    return validate

def training(mel_train_chunked, mel_val_chunked, mel_test_chunked, ParallelModel, loss_fnc, Y_train, Y_val, Y_test, EMOTIONS):
    X_train = np.stack(mel_train_chunked, axis=0)
    X_train = np.expand_dims(X_train, 2)
    print('Shape of X_train: ', X_train.shape)
    X_val = np.stack(mel_val_chunked, axis=0)
    X_val = np.expand_dims(X_val, 2)
    print('Shape of X_val: ', X_val.shape)
    X_test = np.stack(mel_test_chunked, axis=0)
    X_test = np.expand_dims(X_test, 2)
    print('Shape of X_test: ', X_test.shape)

    del mel_train_chunked
    del mel_val_chunked
    del mel_test_chunked

    scaler = StandardScaler()

    b, t, c, h, w = X_train.shape
    X_train = np.reshape(X_train, newshape=(b, -1))
    X_train = scaler.fit_transform(X_train)
    X_train = np.reshape(X_train, newshape=(b, t, c, h, w))

    b, t, c, h, w = X_test.shape
    X_test = np.reshape(X_test, newshape=(b, -1))
    X_test = scaler.transform(X_test)
    X_test = np.reshape(X_test, newshape=(b, t, c, h, w))

    b, t, c, h, w = X_val.shape
    X_val = np.reshape(X_val, newshape=(b, -1))
    X_val = scaler.transform(X_val)
    X_val = np.reshape(X_val, newshape=(b, t, c, h, w))

    EPOCHS = 700
    DATASET_SIZE = X_train.shape[0]
    BATCH_SIZE = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Selected device is {}'.format(device))
    model = ParallelModel(num_emotions=len(EMOTIONS)).to(device)
    print('Number of trainable params: ', sum(p.numel() for p in model.parameters()))
    OPTIMIZER = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.8)

    train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
    validate = make_validate_fnc(model, loss_fnc)
    losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        # schuffle data
        ind = np.random.permutation(DATASET_SIZE)
        X_train = X_train[ind, :, :, :, :]
        Y_train = Y_train[ind]
        epoch_acc = 0
        epoch_loss = 0
        iters = int(DATASET_SIZE / BATCH_SIZE)
        for i in range(iters):
            batch_start = i * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
            actual_batch_size = batch_end - batch_start
            X = X_train[batch_start:batch_end, :, :, :, :]
            Y = Y_train[batch_start:batch_end]
            X_tensor = torch.tensor(X, device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)
            loss, acc = train_step(X_tensor, Y_tensor)
            epoch_acc += acc * actual_batch_size / DATASET_SIZE
            epoch_loss += loss * actual_batch_size / DATASET_SIZE
            print(f"\r Epoch {epoch}: iteration {i}/{iters}", end='')
        X_val_tensor = torch.tensor(X_val, device=device).float()
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.long, device=device)
        val_loss, val_acc, _ = validate(X_val_tensor, Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        print('')
        print(
            f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%")