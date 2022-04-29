import torch
from tqdm import tqdm 
import numpy as np
from sam.sam import SAM


class SAM_Trainer:
    def __init__(self):
        pass

    def compile(self, lr, decay_fn, loss_fn, weight_decay, metric_dict, is_arcface, device): 
        self.lr = lr
        self.decay_fn = decay_fn 
        self.device = device
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.metric_dict = metric_dict
        self.is_arcface = is_arcface

    def _get_optimizer(self, model, lr, weight_decay):
        return SAM(model.parameters(), torch.optim.Adam, lr=lr, weight_decay=weight_decay) 

    def _get_scheduler(self, decay_fn):
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda= decay_fn)

    def fit(self, model, train_loader, validation_loader, num_epoch, save_config, track=False, verbose=True):
        best_score = 0
        self.optimizer = self._get_optimizer(model, self.lr, self.weight_decay)
        self.scheduler = self._get_scheduler(self.decay_fn)
        
        for epoch in range(1, num_epoch+1):
            self._training_step(model, train_loader, self.loss_fn, verbose)
            val_loss, metric_eval_dict = self._validation_step(model, validation_loader, self.loss_fn, self.metric_dict, verbose)

            if epoch % save_config["freq"] == 0:
                torch.save(model, save_config["path"])
            if best_score < metric_eval_dict[save_config["metric"]]:
                torch.save(model, save_config["best_path"])
                best_score = metric_eval_dict[save_config["metric"]]
            if track: 
                self.scheduler.step(val_loss.item())
            else: 
                self.scheduler.step()
        print("best_score:", best_score)

    def _training_step(self, model, train_loader, loss_fn, verbose): 
        model.train()
        loop = tqdm(train_loader) if verbose else train_loader
        total_loss = 0 

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self.device)
            targets = targets.type(torch.LongTensor).to(device=self.device)


            # forward & backward
            for method in ['first_step', 'second_step']: 
                if self.is_arcface: 
                    kwargs = {'target': targets}
                    loss = loss_fn(model(data, **kwargs), targets)
                else: 
                    loss = loss_fn(model(data), targets)

                loss.backward()
                getattr(self.optimizer, method)(zero_grad=True)

                if method == 'first_step': 
                    total_loss += loss.item() * data.size(0)
                    loop.set_postfix(loss=loss.item())            


        if verbose: 
            print("train_loss:", total_loss/len(train_loader.dataset))

    @torch.no_grad()
    def _validation_step(self, model, validation_loader, loss_fn, metric_dict, verbose): 
        model.eval()
        test_loss = 0
        metric_eval_dict = {k:0 for k in metric_dict}

        for data, targets in validation_loader:
            data = data.to(device=self.device)
            targets = targets.type(torch.LongTensor).to(device=self.device)
 
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            test_loss += loss.item() * targets.size(0)

            for metric_name in metric_dict: 
                metric_eval_dict[metric_name] += metric_dict[metric_name](predictions, targets) * targets.size(0)
    
        size = len(validation_loader.dataset)
        test_loss /= size
        metric_eval_dict = {k:(metric_eval_dict[k]/size).item() for k in metric_eval_dict}

        if verbose: 
            print(metric_eval_dict)
        return test_loss, metric_eval_dict
