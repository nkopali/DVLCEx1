import matplotlib.pyplot as plt

class Visualization():

    def __init__(self, epoch, data, title, ylabel):
        self.epoch = epoch
        self.data = data
        self.title = title
        self.ylabel = ylabel
        
    def plot(self):
        plt.plot(self.epoch, self.data,linewidth=2.0)
        plt.xlabel('epoch')
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.show()