
### lr scheduler 

```python
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY)
The learning rate for each epoch is set by multiplying the initial rate by the factor produced by this function.

    rescale_lr = lambda epoch: 1 / (1 + LEARNING_DECAY_RATE * epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda=rescale_lr)
```
