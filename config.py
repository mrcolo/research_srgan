class Config():
    def __init__(self):
        self.dataroot = ""
        self.workers = 4
        self.batchSize = 8
        self.imageSize = 128
        self.upSampling = 4
        self.nEpochs = 50
        self.generatorPretrainEpochs = 3
        self.generatorLR = 0.0001
        self.discriminatorLR = 0.0001
        self.nGPU = 1
        self.resBlocks = 8
        self.generatorWeights = ''
        self.discriminatorWeights = ''
        self.out = "checkpoints"
        self.cuda = False