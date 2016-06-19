import nengo
import nengo_ocl
#from numpy import linalg as LA
import numpy as np
import scipy.spatial as space

model = nengo.Network(label="Difference Target Propagation")


x_step = np.arange(0,2* np.pi,np.pi/16)

def inpfunc(x):
    return np.sin(x)


# Feature vectors of sine. Turns out I didn't need this. The system gets feature
# vectors on its own
feature_vectors = np.transpose(np.asarray([x_step,inpfunc(x_step)]))

# Where to start the second feature vector
hIndex = 100
step = len(x_step)
curstep = 0
n_neurons = 300

#Computes the DTP loss function
def DTP_loss(f, g, x, y):
    epsilon = (np.random.rand(1,2) - 0.5)/16
    return np.power(g + epsilon[0] - x,2) + np.power(f + epsilon[1] - y,2)

def DTP_loss_hidden(f):
    epsilon = (np.random.rand(1,1) - 0.5)/4
    return f + epsilon

def signalDistance(signal):
    if curstep == 0:
        minimum = np.amin([
            space.distance.euclidean(signal,feature_vectors[curstep +1]),
            space.distance.euclidean(signal,feature_vectors[curstep])])

    elif curstep == len(x_step)-1:
        minimum = np.amin([
            space.distance.euclidean(signal,feature_vectors[curstep -1]),
            space.distance.euclidean(signal,feature_vectors[curstep])])

    else:
        minimum = np.amin([
            space.distance.euclidean(signal,feature_vectors[curstep -1]),
            space.distance.euclidean(signal,feature_vectors[curstep]),
            space.distance.euclidean(signal, feature_vectors[curstep+1])])

    return minimum

def cycleinput(x, period, dt=0.01):
    step = int(round(period/dt))
    def stimulus(t):
        i = int(round((t-dt)/dt))
        return x[(i/step)%len(x)]
    return stimulus

x = np.arange(0, 2*np.pi, np.pi/16)
time = (4 * np.pi)
timedelta = time/float(len(x))
import matplotlib.pyplot as plt

with model:
    inputstim = nengo.Node(output=cycleinput(np.transpose([inpfunc(x),x]),timedelta))

    #Set up layers
    in_x = nengo.Ensemble(n_neurons,1)
    h_1 = nengo.Ensemble(n_neurons,1)
    h_2 = nengo.Ensemble(n_neurons,1)
    in_y = nengo.Ensemble(n_neurons,1)


    #Set up the hidden layer and the target ensemble for the first hidden layer
    hiddenC1 = nengo.Connection(in_x, h_1, function = DTP_loss_hidden, learning_rule_type = nengo.PES(1e-3))
    hLoss = nengo.Ensemble(n_neurons,1)
    target = nengo.Ensemble(n_neurons,1)
    target_con = nengo.Connection(target, hLoss, transform=-1)
    target_h_con = nengo.Connection(h_1,target,learning_rule_type = nengo.PES(1e-3))
    nengo.Connection(in_x, hLoss)
    nengo.Connection(hLoss, hiddenC1.learning_rule)

    #Do the same for the second layer
    hiddenC2 = nengo.Connection(h_1, h_2, function = DTP_loss_hidden, learning_rule_type = nengo.PES(1e-3))
    nengo.Connection(h_2, in_y)
    hLoss2 = nengo.Ensemble(n_neurons,1)
    target2 = nengo.Ensemble(n_neurons,1)
    target_con2 = nengo.Connection(target2, hLoss2, transform=-1)
    target_h_con2 = nengo.Connection(h_2,target2,learning_rule_type = nengo.PES(1e-3))
    nengo.Connection(hLoss2, hiddenC2.learning_rule)




    gLoss = nengo.Ensemble(n_neurons,1)
    nengo.Connection(h_2, gLoss)
    #Using input stims since this this would normally be done via a recursive structure
    nengo.Connection(inputstim[0], gLoss, transform=-.5)
    nengo.Connection(in_x, gLoss, function = DTP_loss_hidden)
    nengo.Connection(h_1, gLoss, transform=-1)

    gLoss2 = nengo.Ensemble(n_neurons,1)
    nengo.Connection(in_y, gLoss2)
    #Using input stims since this this would normally be done via a recursive structure

    #f_epsilon = nengo.Ensemble(n_neurons,1)
    #nengo.Connection(h_1, f_epsilon, function=DTP_loss_hidden)
    z_epsilon = nengo.Ensemble(n_neurons,1)
    nengo.Connection(h_2, z_epsilon, function = DTP_loss_hidden)

    nengo.Connection(in_y, gLoss2)
    nengo.Connection(in_x, gLoss2, transform=-1)
    nengo.Connection(inputstim[0], gLoss2, transform = 0.5)
    nengo.Connection(h_2, gLoss, transform=-1)


    #nengo.Connection(gLoss, target_con.learning_rule)
    nengo.Connection(gLoss, target_h_con.learning_rule)
    nengo.Connection(gLoss2, target_h_con2.learning_rule)

    #Input
    nengo.Connection(inputstim[0], in_x, function =lambda x: DTP_loss_hidden(x/5))
    #nengo.Connection(inputstim[1], in_y)

    probe_h1 = nengo.Probe(h_1, 'decoded_output', synapse=0.1)
    probe_h2 = nengo.Probe(h_2, 'decoded_output', synapse=0.1)
    probe_target = nengo.Probe(target, 'decoded_output', synapse = 0.1)
    probe_target2 = nengo.Probe(target2, 'decoded_output', synapse = 0.1)

    probe_x = nengo.Probe(in_x, 'decoded_output', synapse=0.1)
    probe_y = nengo.Probe(in_y, 'decoded_output', synapse=0.1)


    probe_err1 = nengo.Probe(gLoss, synapse=0.1)
    probe_err2 = nengo.Probe(hLoss, synapse=0.1)
    probe_err3 = nengo.Probe(gLoss2, synapse=0.1)
    probe_err4 = nengo.Probe(hLoss2, synapse=0.1)

    probe_stim = nengo.Probe(inputstim, synapse=0.1)
with nengo_ocl.Simulator(model, dt = 0.005) as sim:
    sim.run(time)

#output_xy = np.transpose(sim.data[probe_out_xy])
#First diagram is targets and hidden layers
plt.plot(sim.trange(), sim.data[probe_h1])
plt.plot(sim.trange(), sim.data[probe_h2])
plt.plot(sim.trange(), sim.data[probe_target])
plt.plot(sim.trange(), sim.data[probe_target2])
plt.figure()
#Second diagram is error signals of the first hidden layer
plt.plot(sim.trange(), sim.data[probe_err1])
plt.plot(sim.trange(), sim.data[probe_err2])
plt.figure()
#Error signals of the second hidden layer
plt.plot(sim.trange(), sim.data[probe_err3])
plt.plot(sim.trange(), sim.data[probe_err4])
plt.figure()
#Input and output
plt.plot(sim.trange(), sim.data[probe_x])
plt.plot(sim.trange(), sim.data[probe_y])
plt.show()
