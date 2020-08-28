import torch, time, h5py
import numpy as np
import utils


def forwardKLD(flow, trainLoader, testLoader, epoches, lr, savePeriod, rootFolder, device, eps=1.e-7, warmup=10, lr_decay=0.999, plotfn=None):
    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])

    print('total nubmer of trainable parameters:', nparams)


    # Building gadget for optim
    def lr_lambda(epoch):
        return min(1., (epoch + 1) / warmup) * np.power(lr_decay, epoch)
    optimizer = torch.optim.Adamax(params, lr=lr, eps=eps) # from idf implementation
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1) # from idf implementation
    #optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    LOSS = []
    VALLOSS = []
    bestTrainLoss = 99999999
    bestTestLoss = 99999999

    for e in range(epoches):
        print(" Training " + str(e + 1) + "-th epoch")

        # train
        trainLoss = []
        t_start = time.time()
        for samples, _ in trainLoader:
            samples = samples.to(device)
            lossRaw = -flow.logProbability(samples)
            _loss = lossRaw.mean()

            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()

            trainLoss.append(_loss.detach().cpu().item())
        trainLoss = np.array(trainLoss)
        trainTime = time.time() - t_start
        LOSS.append(trainLoss.mean())

        # vaildation
        testLoss = []
        for samples, _ in testLoader:
            samples = samples.to(device)
            lossRaw = -flow.logProbability(samples)
            _loss = lossRaw.mean()
            testLoss.append(_loss.detach().cpu().item())
        testLoss = np.array(testLoss)
        LOSS.append(testLoss.mean())

        # step the optimizer scheduler
        scheduler.step()

        # feedback
        print("Train time:", trainTime)
        print("Mean train loss:", trainLoss.mean(), "Mean vaildation loss:", testLoss.mean())
        print("Best train loss:", trainLoss.min(), "Best vaildation loss:", testLoss.min())

        # save
        if e % savePeriod == 0:
            torch.save(flow, rootFolder + 'savings/' + flow.name + "_epoch_" + str(e) + ".saving")
            torch.save(optimizer, rootFolder + 'savings/' + flow.name + "_epoch_" + str(e) + "opt.saving")
            with h5py.File(rootFolder + "records/" + "epoch_" + str(e) + '.hdf5', 'w') as f:
                f.create_dataset("LOSS", data=np.array(LOSS))
                f.create_dataset("VALLOSS", data=np.array(VALLOSS))

            # select best model
            if trainLoss.mean() < bestTrainLoss:
                bestTrainLoss = trainLoss.mean()
                torch.save(flow, rootFolder + 'best_TrainLoss_model.saving')
            if testLoss.mean() < bestTestLoss:
                bestTestLoss = testLoss.mean()
                torch.save(flow, rootFolder + 'best_TestLoss_model.saving')

            # plot
            if plotfn is not None:
                plotfn(flow, trainLoader, testLoader, LOSS, VALLOSS)
            utils.cleanSaving(rootFolder, e, 6 * savePeriod, flow.name)

    return flow