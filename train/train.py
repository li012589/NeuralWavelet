import torch, time, h5py
import numpy as np
import utils


def gradTest(params):
    zeroCount = 0
    nonCount = 0
    for no, param in enumerate(params):
        if param.grad is None:
            nonCount += 1
        elif torch.abs(param.grad).sum() == 0:
            zeroCount += 1
    return zeroCount, nonCount


def forwardKLD(flow, trainLoader, testLoader, epoches, lr, savePeriod, rootFolder, eps=1.e-7, warmup=10, lr_decay=0.999, plotfn=None, testGrad=False):
    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])

    print('total nubmer of trainable parameters:', nparams)
    print('parameter groups:', len(params))

    # Building gadget for optim
    def lr_lambda(epoch):
        return min(1., (epoch + 1) / warmup) * np.power(lr_decay, epoch)
    optimizer = torch.optim.Adamax(params, lr=lr, eps=eps) # from idf implementation
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1) # from idf implementation
    #optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    LOSS = []
    trainBPD = []
    testBPD = []
    VALLOSS = []
    bestTrainLoss = 99999999
    bestTestLoss = 99999999

    for e in range(epoches):
        print(" Training " + str(e + 1) + "-th epoch")

        # train
        trainLoss = []
        t_start = time.time()
        if testGrad:
            countList = []
        for samples, _ in trainLoader:
            lossRaw = -flow.logProbability(samples)
            _loss = lossRaw.mean()

            optimizer.zero_grad()
            _loss.backward()
            if testGrad:
                zeroCounter, nonCounter = gradTest(params)
                countList.append((zeroCounter, nonCounter))
            optimizer.step()

            trainLoss.append(_loss.detach().cpu().item() / (np.prod(samples.shape[1:]) * np.log(2.)))
        if testGrad:
            print("grad test: (zero, none):", countList)
        trainLoss = np.array(trainLoss)
        trainTime = time.time() - t_start
        LOSS.append(trainLoss.mean())
        meanTrainBpd = trainLoss.mean()
        trainBPD.append(meanTrainBpd)

        # vaildation
        testLoss = []
        for samples, _ in testLoader:
            with torch.no_grad():
                lossRaw = -flow.logProbability(samples)
                _loss = lossRaw.mean()
            testLoss.append(_loss.detach().cpu().item() / (np.prod(samples.shape[1:]) * np.log(2.)))
        testLoss = np.array(testLoss)
        VALLOSS.append(testLoss.mean())
        meanTestBpd = testLoss.mean()
        testBPD.append(meanTestBpd)


        # step the optimizer scheduler
        scheduler.step()

        # feedback
        print("Train time:", trainTime)
        print("Mean train loss:", trainLoss.mean(), "Mean vaildation loss:", testLoss.mean())
        print("Mean train bpd:", meanTrainBpd, "Mean vaildation bpd:", meanTestBpd)
        print("Best train loss:", min(LOSS), "Best vaildation loss:", min(VALLOSS))
        print("Best train bdp:", min(trainBPD), "Best vaildation loss:", min(testBPD))
        print("====================================================================")

        # select best model
        if trainLoss.mean() < bestTrainLoss:
            bestTrainLoss = trainLoss.mean()
            torch.save(flow, rootFolder + 'best_TrainLoss_model.saving')
        if testLoss.mean() < bestTestLoss:
            bestTestLoss = testLoss.mean()
            torch.save(flow, rootFolder + 'best_TestLoss_model.saving')

        # save
        if e % savePeriod == 0:
            torch.save(flow, rootFolder + 'savings/' + flow.name + "_epoch_" + str(e) + ".saving")
            torch.save(optimizer, rootFolder + 'savings/' + flow.name + "_epoch_" + str(e) + "_opt.saving")
            with h5py.File(rootFolder + "records/" + "LOSSES" + '.hdf5', 'w') as f:
                f.create_dataset("LOSS", data=np.array(LOSS))
                f.create_dataset("VALLOSS", data=np.array(VALLOSS))

            # plot
            if plotfn is not None:
                plotfn(flow, trainLoader, testLoader, LOSS, VALLOSS)
            utils.cleanSaving(rootFolder, e, 6 * savePeriod, flow.name)

    return flow