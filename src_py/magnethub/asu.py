# Inference script *accept temp 25,50,70,90 only
# one material at each call, not batched
# Nov-1 (10 material model references)
# quick checked 
#      + with https://asumag.streamlit.app/ 
#      + self-hosted : results from paderborn and sydney)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy 
import torch
import torch.nn as nn
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

MAT2FILENAME = {
    "3C90": "Model3C90.sd",
    "3C94": "Model3C94.sd",
    "3E6": "Model3E6.sd",
    "3F4": "Model3F4.sd",
    "77": "Model77.sd",
    "78": "Model78.sd",
    "79": "Model79.sd",
    "N27": "ModelN27.sd",
    "N30": "ModelN30.sd",
    "N49": "ModelN49.sd",
    "N87": "ModelN87.sd",
}
nparams = 24

def nearest_value(value,arr=np.array([25,50,70,90])):
    index = (np.abs(arr - value)).argmin()
    nearest_value = arr[index]
    return nearest_value
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(nparams+5,nparams+5),
            nn.ReLU(),
            nn.Linear(nparams+5,15),
            nn.ReLU(),
            nn.Linear(15,15),
            nn.ReLU(),
            nn.Linear(15, 1),
        )

    def forward(self, x):
        """
        """
        return self.layers(x)

def model_run(net,waveshape,frequency,temperature):
    device = torch.device("cpu")
    if isinstance(frequency,np.ndarray):
        pass
    else:
        frequency=np.array([[frequency]])
    if isinstance(temperature,np.ndarray):
        pass
    else:
        temperature=np.array([[temperature]])
    vectorized_nearest_value = np.vectorize(nearest_value)

    temperature = vectorized_nearest_value(temperature)
    waveshape=waveshape.reshape((-1,1024))
    fft_data = np.fft.fft(waveshape, axis=1)
    fft_data=scipy.fft.ifft(fft_data[:, :nparams],n=nparams, axis=1)
    Flux =np.abs(fft_data) 
    Freq = np.log10(frequency)
    Flux = np.log10(Flux)
    
    # Reshape data
    Freq = Freq.reshape((-1,1))
    temperature = temperature.reshape((-1,1))

    Flux = Flux.reshape((-1,nparams))
    enc = preprocessing.OneHotEncoder()
    # 2. FIT
    enc.fit([[25],[50],[70],[90]])
    # 3. Transform
    T = enc.transform(temperature).toarray().reshape((-1,4))

    temp = np.concatenate((Freq,Flux,T),axis=1)
    inputs = torch.from_numpy(temp).view(-1, nparams + 1 +T.shape[1])

    with torch.no_grad():
        y_pred=net(inputs.to(device))

    return 10**y_pred.numpy()
def one_sample_predict(material,waveshape,frequency,temperature):
    nparams=24
    device = torch.device("cpu")    

    net = Net().double().to(device)

    net.load_state_dict(torch.load(f"models/asu/Model{material}.sd"))
    net.eval()

    return model_run(net,waveshape,frequency,temperature)

    
class AsuModel:
    """The ASU model.

    E. Havugimana and M. Ranjram, “A Simple Data-Driven Machine Learning-based 
    Software Package for Accurate Magnetic Core Loss Computation,” Oct. 2025.
    """
    
    expected_seq_len = 1024  # the expected sequence length
    def __init__(self, model_path, material):

        self.model_path = model_path
        self.material = material
        self.device = torch.device("cpu")    
        net=Net().double().to(self.device)
        print(model_path)
        net.load_state_dict(torch.load(model_path))
        self.model=net
        assert (
            material in MAT2FILENAME.keys()
        ), f"Requested material '{material}' is not supported"
        self.predicts_p_directly = True

    def __call__(self, b_seq, frequency, temperature):
        """Evaluate trajectory and estimate power loss.

        Args
        ----
        b_seq: (X, Y) array_like
            The magnetic flux density array(s) in T. First dimension X describes the batch, the second Y
             the time length (will always be interpolated to 1024 samples)
        frequency: scalar or 1D array-like
            The frequency operation point(s) in Hz
        temperature: scalar or 1D array-like
            The temperature operation point(s) in °C
            Closest temperature to 4 temperatures in magnet dataset is used.
            Future interpolation changes planned

        Return
        ------
        p, h: (X,) np.array, None
            The estimated power loss (p) in W/m³ and  h is None
        """
        self.model.eval()
        p_pred=model_run(self.model,b_seq,frequency,temperature)
        h_pred = None
        p_pred=p_pred.astype(np.float32)
        return  p_pred.reshape(-1,) , None

if __name__ == "__main__":
    material="N30"
    waveshape=0.1*np.sin(np.linspace(-np.pi/2,2*np.pi-np.pi/2,1024)).reshape(1,1024)
    waveshape=np.vstack((waveshape,waveshape))
    temperature=np.array([[25],[90]])
    frequency=np.array([[100e3],[100e3]])
    pred=one_sample_predict(material,waveshape,frequency,temperature)
    print(pred/1e3,"kW/m^3")
    # plt.show()
    mdl = AsuModel("models/asu/ModelN87.sd",material="N87")

    # dummy B field data (one trajectory with 1024 samples)
    b_wave = np.random.randn(1024)* 200e-3  # in T
    freq = 124062  # Hz
    temp = 25  # °C

    # get power loss in W/m³ and estimated H wave in A/m
    p, h = mdl(b_wave, freq, temp)
    print(p)
    # batch execution for 100 trajectories
    b_waves = np.random.randn(100, 1024)* 200e-3  # in T
    freqs = np.random.randint(100e3, 750e3, size=100)
    temps = np.random.randint(20, 80, size=100)
    # temps=[25]*len(temps)
    p, h = mdl(b_waves, freqs, temps)
    print(p)
