import keras.callbacks as callbacks
import math, copy, random, os
import util.Metrics as Metrics
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


class CosineLRSchedule:
    """
    Cosine annealing with warm restarts, described in paper
    "SGDR: stochastic gradient descent with warm restarts"
    https://arxiv.org/abs/1608.03983

    Changes the learning rate, oscillating it between `lr_high` and `lr_low`.
    It takes `period` epochs for the learning rate to drop to its very minimum,
    after which it quickly returns back to `lr_high` (resets) and everything
    starts over again.

    With every reset:
        * the period grows, multiplied by factor `period_mult`
        * the maximum learning rate drops proportionally to `high_lr_mult`

    This class is supposed to be used with
    `keras.callbacks.LearningRateScheduler`.
    """
    def __init__(self, lr_high: float, lr_low: float, initial_period: int = 50,
                 period_mult: float = 2, high_lr_mult: float = 0.97):
        self._lr_high = lr_high
        self._lr_low = lr_low
        self._initial_period = initial_period
        self._period_mult = period_mult
        self._high_lr_mult = high_lr_mult

    def __call__(self, epoch, lr):
        return self.get_lr_for_epoch(epoch)

    def get_lr_for_epoch(self, epoch):
        assert epoch >= 0
        t_cur = 0
        lr_max = self._lr_high
        period = self._initial_period
        result = lr_max
        for i in range(epoch + 1):
            if i == epoch:  # last iteration
                result = (self._lr_low +
                          0.5 * (lr_max - self._lr_low) *
                          (1 + math.cos(math.pi * t_cur / period)))
            else:
                if t_cur == period:
                    period *= self._period_mult
                    lr_max *= self._high_lr_mult
                    t_cur = 0
                else:
                    t_cur += 1
        return result


def fit(model, train_generator, valid_generator, epoch_limit=200, loss_nonimprove_limit=3, batch_size=32, use_cosine_lr=True, model_save_path=None):
    from model.multihot_utils import recall_at_10
    # Train model with early stopping condition
    metric = 'recall_at_10'
    print('Training model...')
    base_logger = callbacks.BaseLogger(stateful_metrics=['recall_at_10', 'val_recall_at_10'])
    early_stopping = callbacks.EarlyStopping(monitor='val_recall_at_10', patience=loss_nonimprove_limit, verbose=1, mode='max')
    model_callbacks = [base_logger, early_stopping]

    if use_cosine_lr:
        model_callbacks.append(callbacks.LearningRateScheduler(
        CosineLRSchedule(lr_high=1e-4, lr_low=1e-4 / 32, initial_period=10), verbose=1))
    if model_save_path is not None:
        model_callbacks.append(callbacks.ModelCheckpoint(model_save_path, monitor='val_recall_at_10', mode='max', save_best_only=True, verbose=True))

    model_history = model.fit_generator(
        generator=train_generator, 
        validation_data=valid_generator,
        epochs=epoch_limit, 
        callbacks=model_callbacks,
        use_multiprocessing=True,
        workers=5)

    best_accuracy = max(model_history.history[metric])
    print("Best accuracy:", best_accuracy)
    

def test(model, generator, pred_window):
    target_list, predict_list = [], []
    for iter, batch in enumerate(tqdm(generator, ncols=60)):
        if iter == len(generator): break
        
        predict = model.predict_on_batch(batch[0])[0]
        
        target_list.append(batch[0][-1][:, pred_window[0]:pred_window[1]])
        predict_list.append(predict[:, pred_window[0]:pred_window[1]])
    target = np.concatenate(target_list, axis=0)
    predict = np.concatenate(predict_list, axis=0)
    recall, recall_per_sem = Metrics.recall(target, predict, at_n=10)
    print('Recall: {:.4f}'.format(recall))
    print('Recall per Semester: {}'.format(recall_per_sem.round(4)))
    return recall, recall_per_sem


def test_auto_regressive(model, generator, pred_window, top_n):
    target_list, predict_list = [], []
    for iter, batch in enumerate(tqdm(generator, ncols=60)):
        if iter == len(generator): break
        
        input = batch[0]
        for t in range(0, pred_window[1] - pred_window[0]):
            predict = model.predict_on_batch(input)[0]
            if top_n:
                rank = np.argsort(predict[:, pred_window[0] + t], axis=-1)[:, -10:]
                for u in range(predict.shape[0]):
                    input[0][u, pred_window[0] + t, rank[u]] = 1
            else:
                input[0][:, pred_window[0] + t] = predict[:, pred_window[0] + t]
                
        predict = model.predict_on_batch(batch[0])[0]
        target_list.append(batch[0][-1][:, pred_window[0]:pred_window[1]])
        predict_list.append(predict[:, pred_window[0]:pred_window[1]])
    target = np.concatenate(target_list, axis=0)
    predict = np.concatenate(predict_list, axis=0)
    recall, recall_per_sem = Metrics.recall(target, predict, at_n=10)
    print('Recall: {:.4f}'.format(recall))
    print('Recall per Semester: {}'.format(recall_per_sem.round(4)))
    return recall, recall_per_sem


def wishlist_test_onehot(model, generator, pred_window):
    target_list, predict_list = [], []
    for iter, batch in enumerate(tqdm(generator, ncols=60)):
        if iter == len(generator): break
            
        wish_list = batch[0][0][:, pred_window[0]:pred_window[1]].sum(1)
        batch[0][0][:, pred_window[0]:pred_window[1]] = 0
        predict = model.predict_on_batch(batch[0])[0][:, pred_window[0]:pred_window[1]]
        rank = predict.argsort(1)[:, -1]
        for user in range(generator.batch_size):
            temp_wish_list = np.where(wish_list[user])[0]
            batch[0][0][user, pred_window[0] + rank[user, temp_wish_list], temp_wish_list] = 1
            
        predict = model.predict_on_batch(batch[0])[0]
        target_list.append(batch[0][-1][:, pred_window[0]:pred_window[1]])
        predict_list.append(predict[:, pred_window[0]:pred_window[1]])
    target = np.concatenate(target_list, axis=0)
    predict = np.concatenate(predict_list, axis=0)
    recall, recall_per_sem = Metrics.recall(target, predict, at_n=10)
    print('Recall: {:.4f}'.format(recall))
    print('Recall per Semester: {}'.format(recall_per_sem.round(4)))
    return recall, recall_per_sem


def wishlist_test_possibility(model, generator, pred_window):
    target_list, predict_list = [], []
    for iter, batch in enumerate(tqdm(generator, ncols=60)):
        if iter == len(generator): break
            
        wish_list = np.where(batch[0][0][:, pred_window[0]:pred_window[1]].sum(1))[0]
        batch[0][0][:, pred_window[0]:pred_window[1]] = 0
        predict = model.predict_on_batch(batch[0])[0][:, pred_window[0]:pred_window[1]]
        
        future = predict[:, pred_window[0]:pred_window[1], wish_list]
        future = future / future.sum(1, keepdims=True)
        batch[0][0][:, pred_window[0]:pred_window[1], wish_list] = future
        
        predict = model.predict_on_batch(batch[0])[0]
        target_list.append(batch[0][-1][:, pred_window[0]:pred_window[1]])
        predict_list.append(predict[:, pred_window[0]:pred_window[1]])
    target = np.concatenate(target_list, axis=0)
    predict = np.concatenate(predict_list, axis=0)
    recall, recall_per_sem = Metrics.recall(target, predict, at_n=10)
    print('Recall: {:.4f}'.format(recall))
    print('Recall per Semester: {}'.format(recall_per_sem.round(4)))
    return recall, recall_per_sem

    
def GPU_max_free_memory():
    import pynvml  
    pynvml.nvmlInit()
    free_list = []
    for iter in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(iter)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_list.append(meminfo.free)
    return np.argmax(free_list)