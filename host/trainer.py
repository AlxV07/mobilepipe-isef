import json
import queue
import threading

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mobilepipe.comm.comm_utils import CommHandler
from mobilepipe.train.timekeeping import TimeKeeper


class TrainingArguments:
    def __init__(self, learning_rate: float, epochs: int, batch_size: int, microbatch_size: int = None, nof_dynamic_batches: int = None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.microbatch_size = microbatch_size
        self.nof_dynamic_batches = nof_dynamic_batches


class DefaultTrainer:
    def __init__(self, model, args: TrainingArguments, train_dataset):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset

    def train(self):
        TIMEK = TimeKeeper()
        global_step = 0
        device = torch.device('cpu')

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True)

        TIMEK.start(TIMEK.CATS.TRAINING_RUN)
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")

            for step, batch in enumerate(train_loader):
                TIMEK.start(TIMEK.CATS.FULL_BATCH)
                batch: dict
                inputs, targets = map(lambda t: t[1].to(device), batch.items())

                optimizer.zero_grad()

                logits = self.model.forward(inputs)
                loss = nn.functional.cross_entropy(logits, targets)

                loss.backward()
                optimizer.step()

                TIMEK.end(TIMEK.CATS.FULL_BATCH)

                global_step += 1

                start, end = TIMEK.times[TIMEK.CATS.FULL_BATCH][-1].get_time()
                print(f"epoch {epoch + 1} | step {step}/{len(train_loader)} | loss = {loss.item():.4f} | {global_step} / {self.args.epochs * len(train_loader)} | {end - start:.4f}s")
        TIMEK.end(TIMEK.CATS.TRAINING_RUN)
        TIMEK.write_to_file('Default_training.json')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


class MobilePipeTrainer:
    """
    mobilepipe Trainer class for experiments
    """

    def __init__(self, model, args: TrainingArguments, train_dataset, experiment_id):
        self.comm_handler = CommHandler()
        self.comm_handler.connect(default=True)
        self.experiment_id = experiment_id
        self.microbatch_size = args.microbatch_size

        self.model = model
        self.args = args

        self.train_dataset = train_dataset

    def train(self):
        self._train_dynamic_resnet()

    def _dynamic_resnet34_pipeline_s1_configs(self):
        """
        :return: G1 (ends L1), G2 (ends L2), G3 (ends L3)
        """
        G1_s1 = nn.Sequential(self.model.conv1,
                              self.model.bn1,
                              self.model.relu,
                              self.model.maxpool,
                              self.model.layer1)
        G2_s1 = nn.Sequential(G1_s1,
                              self.model.layer2)
        G3_s1 = nn.Sequential(G2_s1,
                              self.model.layer3)
        return G1_s1, G2_s1, G3_s1

    def _ios_parameters(self):
        """
        :return: List of all 147 which the iOS stage might potentially manage (Layer2 +)
        """

        # 147 params
        m = self.model
        res = []

        # layer2
        # First block (index 0) has downsample
        bn1 = m.layer2[0].bn1
        bn2 = m.layer2[0].bn2
        bn = m.layer2[0].downsample[1]
        res.extend([
            m.layer2[0].conv1.weight, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var,
            m.layer2[0].conv2.weight, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var,
            m.layer2[0].downsample[0].weight, bn.weight, bn.bias, bn.running_mean, bn.running_var,
        ])
        # Remaining blocks (indices 1-3) don't have downsample
        for block in range(1, 4):
            bn1 = m.layer2[block].bn1
            bn2 = m.layer2[block].bn2
            res.extend([
                m.layer2[block].conv1.weight, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var,
                m.layer2[block].conv2.weight, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var,
            ])
        # layer3
        bn1 = m.layer3[0].bn1
        bn2 = m.layer3[0].bn2
        bn = m.layer3[0].downsample[1]
        res.extend([
            m.layer3[0].conv1.weight, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var,
            m.layer3[0].conv2.weight, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var,
            m.layer3[0].downsample[0].weight, bn.weight, bn.bias, bn.running_mean, bn.running_var,
        ])
        for block in range(1, 6):
            bn1 = m.layer3[block].bn1
            bn2 = m.layer3[block].bn2
            res.extend([
                m.layer3[block].conv1.weight, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var,
                m.layer3[block].conv2.weight, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var,
            ])
        # layer4
        bn1 = m.layer4[0].bn1
        bn2 = m.layer4[0].bn2
        bn = m.layer4[0].downsample[1]
        res.extend([
            m.layer4[0].conv1.weight, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var,
            m.layer4[0].conv2.weight, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var,
            m.layer4[0].downsample[0].weight, bn.weight, bn.bias, bn.running_mean, bn.running_var,
        ])
        for block in range(1, 3):
            bn1 = m.layer4[block].bn1
            bn2 = m.layer4[block].bn2
            res.extend([
                m.layer4[block].conv1.weight, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var,
                m.layer4[block].conv2.weight, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var,
            ])
        # fc
        res.extend([m.fc.weight, m.fc.bias])
        return res

    def _train_dynamic_resnet(self):
        device = torch.device('cpu')

        self.model.to(device)
        self.model.train()

        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      drop_last=True)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)

        TIMEK = TimeKeeper()

        TIMEK.start(TIMEK.CATS.INIT_MODEL_OFFLOAD)
        self.comm_handler.send_all_parameters(list(map(lambda p: p.detach(), self._ios_parameters())))
        TIMEK.end(TIMEK.CATS.INIT_MODEL_OFFLOAD)
        self.comm_handler.send_input(self.experiment_id)

        print('Starting MobilePipe training...')

        train_loader_iter = iter(enumerate(train_dataloader))

        self.comm_handler.send_uint8(self.args.nof_dynamic_batches)  # send nof dynamic test batches
        self.comm_handler.send_uint8(self.args.batch_size // self.microbatch_size)  # send nof microbatches
        self.comm_handler.send_uint8(self.args.batch_size // 32)  # send scale_lr

        NOF_RESNET34_PARAMS, NOF_RESNET34_L2_PARAMS, NOF_RESNET34_L3_PARAMS = 147, 45, 65
        C1, C2, C3 = self._dynamic_resnet34_pipeline_s1_configs()
        stage1 = C1

        def stage1_forward(_input_microbatches, s1, _activations, _target_microbatches):
            for _idx, input_microbatch in enumerate(_input_microbatches):
                TIMEK.start(TIMEK.CATS.S1_FORWARD_MICROBATCH)
                _stage1_activation = s1(input_microbatch)
                _activations.append(_stage1_activation)
                forward_q.put(_idx)
                TIMEK.end(TIMEK.CATS.S1_FORWARD_MICROBATCH)
        
        def send_queuing(_forward_q, _target_microbatches, _activations, comm_handler):
            for _idx, target_microbatch in enumerate(_target_microbatches):
                _stage1_activation: torch.Tensor = _activations[_forward_q.get()]
                comm_handler.send_tensor(_stage1_activation.detach())
                comm_handler.send_tensor(target_microbatch.detach().to(torch.float32))

        def stage2_forward_and_backward_queuing(_target_microbatches, comm_handler, _forward_q, _backward_q, _activations):
            # running in a separate thread
            for _idx, target_microbatch in enumerate(_target_microbatches):
                received_grad = comm_handler.receive_tensor().to(device)
                received_loss = comm_handler.receive_tensor().to(device)
                _backward_q.put((received_grad, received_loss))

        # === DYNAMIC CONFIGURATION EVALUATION ===
        for CUR_CONFIG_TEST_IDX in range(3):
            print(f'===== STARTING CONFIG TEST {CUR_CONFIG_TEST_IDX + 1}')
            for _ in range(self.args.nof_dynamic_batches):
                TIMEK.start(TIMEK.CATS.FULL_BATCH)
                try:
                    step, batch = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(enumerate(train_dataloader))
                    step, batch = next(train_loader_iter)
                optimizer.zero_grad()

                forward_q = queue.Queue()
                backward_q = queue.Queue()
                activations = []

                # prepare microbatches
                inputs, targets = map(lambda it: it[1], batch.items())
                input_microbatches = inputs.to(device).split(self.microbatch_size)
                target_microbatches = targets.to(device).split(self.microbatch_size)
                self.comm_handler.send_uint8(len(input_microbatches))  # send nof microbatches

                thread = threading.Thread(target=lambda: stage2_forward_and_backward_queuing(target_microbatches,
                                                                                             self.comm_handler,
                                                                                             forward_q, backward_q, activations))
                thread2 = threading.Thread(target = lambda: send_queuing(forward_q, target_microbatches, activations, self.comm_handler))
                thread2.start()
                thread.start()

                # forward
                stage1_forward(input_microbatches, stage1, activations, target_microbatches)

                # backward steps
                for idx, stage1_activation in enumerate(activations):
                    TIMEK.start(TIMEK.CATS.WAITING_FOR_GRAD)
                    gradient, loss = backward_q.get()
                    TIMEK.end(TIMEK.CATS.WAITING_FOR_GRAD)

                    TIMEK.start(TIMEK.CATS.S1_BACKWARD_MICROBATCH)
                    stage1_activation.backward(gradient, retain_graph=idx < len(activations) - 1)
                    TIMEK.end(TIMEK.CATS.S1_BACKWARD_MICROBATCH)

                TIMEK.start(TIMEK.CATS.OPTIMIZATION_STEP)
                # optimizer.step()  # WARMUP: not actually updating weights
                TIMEK.end(TIMEK.CATS.OPTIMIZATION_STEP)
                TIMEK.end(TIMEK.CATS.FULL_BATCH)

                thread2.join()
                thread.join()

            stage1 = C2 if CUR_CONFIG_TEST_IDX == 0 else C3

            # receive ios stage times
            TIMEK.receive_ios_times(comm_handler=self.comm_handler)
            TIMEK.write_to_file(f'MobilePipe_C{CUR_CONFIG_TEST_IDX + 1}.json')
            TIMEK.clear()

        # determine fastest dynamic pipeline configuration
        fastest = None
        best_avg_time_per_batch = float('inf')
        for i in range(1, 4):
            with open(f'MobilePipe_C{i}.json') as f:
                times = json.load(f)

                total_time = 0
                nof_full_batches = len(times['HOST_FULL_BATCH'])

                for j in range(nof_full_batches):
                    start, end = times['HOST_FULL_BATCH'][j]
                    total_time += end - start
                cur_avg_time_per_batch = total_time / nof_full_batches

                if cur_avg_time_per_batch < best_avg_time_per_batch:
                    best_avg_time_per_batch = cur_avg_time_per_batch
                    fastest = i
                print(i, cur_avg_time_per_batch)

        print('Selected:', fastest)

        self.comm_handler.send_uint8(fastest)  # not done
        if fastest != 3:
            for idx, param in enumerate(self._ios_parameters(), start=1):
                if idx > NOF_RESNET34_L2_PARAMS + NOF_RESNET34_L3_PARAMS:
                    break
                if idx % 5 - 1 != 0 or (fastest == 2 and idx <= NOF_RESNET34_L2_PARAMS):
                    continue
                self.comm_handler.send_uint8(idx)  # not done
                self.comm_handler.send_tensor(param.detach())
        self.comm_handler.send_uint8(0)  # done
        stage1 = C1 if fastest == 1 else C2 if fastest == 2 else C3

        # === MAIN TRAINING ===
        print('Starting main training...')
        global_step = 0
        TIMEK.start(TIMEK.CATS.TRAINING_RUN)
        for epoch in range(int(self.args.epochs)):
            for step, batch in enumerate(train_dataloader):
                TIMEK.start(TIMEK.CATS.FULL_BATCH)
                optimizer.zero_grad()

                forward_q = queue.Queue()
                backward_q = queue.Queue()
                activations = []

                # prepare microbatches
                batch: dict
                inputs, targets = map(lambda it: it[1], batch.items())
                input_microbatches = inputs.to(device).split(self.microbatch_size)
                target_microbatches = targets.to(device).split(self.microbatch_size)
                self.comm_handler.send_uint8(len(input_microbatches))  # send nof microbatches

                thread = threading.Thread(target=lambda: stage2_forward_and_backward_queuing(target_microbatches,
                                                                                             self.comm_handler,
                                                                                             forward_q, backward_q, activations))
                thread2 = threading.Thread(target = lambda: send_queuing(forward_q, target_microbatches, activations, self.comm_handler))
                thread2.start()
                thread.start()

                # forward
                stage1_forward(input_microbatches, stage1, activations, target_microbatches)

                # backward steps
                batch_loss = 0
                for idx, stage1_activation in enumerate(activations):
                    TIMEK.start(TIMEK.CATS.WAITING_FOR_GRAD)
                    gradient, loss = backward_q.get()
                    TIMEK.end(TIMEK.CATS.WAITING_FOR_GRAD)

                    TIMEK.start(TIMEK.CATS.S1_BACKWARD_MICROBATCH)
                    stage1_activation.backward(gradient, retain_graph=idx < len(activations) - 1)
                    TIMEK.end(TIMEK.CATS.S1_BACKWARD_MICROBATCH)
                    batch_loss += loss.item()  # loss is already scaled on iOS
                batch_loss /= len(activations)

                TIMEK.start(TIMEK.CATS.OPTIMIZATION_STEP)
                optimizer.step()
                TIMEK.end(TIMEK.CATS.OPTIMIZATION_STEP)

                global_step += 1
                TIMEK.end(TIMEK.CATS.FULL_BATCH)

                start, end = TIMEK.times[TIMEK.CATS.FULL_BATCH][-1].get_time()
                print(f"epoch {epoch + 1} | step {step}/{len(train_dataloader)} | loss = {batch_loss:.4f} | global_step = {global_step}/{len(train_dataloader) * self.args.epochs} | {end - start}")

                thread2.join()
                thread.join()

        TIMEK.end(TIMEK.CATS.TRAINING_RUN)

        TIMEK.write_to_file('MobilePipe_training.json')

        self.comm_handler.send_uint8(0)  # send finished

        # final sync
        params = self._ios_parameters()
        while True:
            i = int(self.comm_handler.receive_double())
            if i == 0:
                break
            t: torch.Tensor = params[i - 1]  # id -> idx
            with torch.no_grad():
                t.copy_(self.comm_handler.receive_tensor())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

