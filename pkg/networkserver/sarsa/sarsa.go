package sarsa

import (
	"fmt"
	"math/rand"

	"go.thethings.network/lorawan-stack/v3/pkg/ttnpb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Constants
const (
	numActions         = 16 // Data rate values from 0 to 15
	alpha              = 0.1
	gamma              = 0.9
	epsilon            = 0.1
	numEpisodes        = 1000
	maxStepsPerEpisode = 100
	batchSize          = 32
	memorySize         = 1000
	gatewayload        = 20
)

// Device represents a LoRaWAN end device
type Device struct {
	Locationx           int
	Locationy           int
	SNR                 float32
	CurrentDataRate     ttnpb.DataRateIndex
	CurrentDemodFloor   float64
	CurrentBatteryLevel float64 //convert to energy levels ************
	ChannelSteering     bool
}

var (
	numDevices = 10
	numStates  = 16 // Assuming we discretize the combined state space into 16 states for simplicity
	nn         *NeuralNetwork
	memory     []Experience
)

// Experience represents a SARSA experience tuple
type Experience struct {
	State      []float64 //comments************
	Action     ttnpb.DataRateIndex
	Reward     float64
	NextState  []float64
	NextAction ttnpb.DataRateIndex
}

// NeuralNetwork represents a simple neural network for Q-value approximation
type NeuralNetwork struct {
	g          *gorgonia.ExprGraph
	w0, w1     *gorgonia.Node
	pred       *gorgonia.Node
	vmachine   gorgonia.VM
	solver     *gorgonia.AdamSolver
	stateSize  int
	actionSize int
}

// NewNeuralNetwork creates a new neural network for Q-value approximation
func NewNeuralNetwork(stateSize, actionSize int) *NeuralNetwork {
	g := gorgonia.NewGraph()
	w0 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(stateSize, 128), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotU(1)))
	w1 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(128, actionSize), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotU(1)))
	return &NeuralNetwork{
		g:          g,
		w0:         w0,
		w1:         w1,
		vmachine:   gorgonia.NewTapeMachine(g),
		solver:     gorgonia.NewAdamSolver(gorgonia.WithLearnRate(alpha)),
		stateSize:  stateSize,
		actionSize: actionSize,
	}
}

// Predict predicts the Q-values for a given state
func (nn *NeuralNetwork) Predict(state []float64) ([]float64, error) {

	x := gorgonia.NewTensor(nn.g, tensor.Float32, 1, gorgonia.WithShape(1, nn.stateSize), gorgonia.WithValue(tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(1, nn.stateSize), tensor.WithBacking(state))))
	l0 := gorgonia.Must(gorgonia.Mul(x, nn.w0))
	a0 := gorgonia.Must(gorgonia.Rectify(l0))
	l1 := gorgonia.Must(gorgonia.Mul(a0, nn.w1))
	nn.pred = l1
	if err := nn.vmachine.RunAll(); err != nil {
		return nil, err
	}
	defer nn.vmachine.Reset()

	output := nn.pred.Value().Data().([]float32)
	return float32ToFloat64(output), nil
}

// Train trains the neural network using the given batch of experiences
func (nn *NeuralNetwork) Train(batch []Experience) error {
	states := make([]float32, len(batch)*nn.stateSize)
	targets := make([]float32, len(batch)*nn.actionSize)
	for i, exp := range batch {
		copy(states[i*nn.stateSize:(i+1)*nn.stateSize], float64ToFloat32(exp.State))
		qValues, _ := nn.Predict(exp.State)
		nextQValues, _ := nn.Predict(exp.NextState)
		target := qValues[exp.Action] + alpha*(exp.Reward+gamma*nextQValues[exp.NextAction]-qValues[exp.Action])
		copy(targets[i*nn.actionSize:(i+1)*nn.actionSize], float64ToFloat32(qValues))
		targets[i*nn.actionSize+int(exp.Action)] = float32(target)
	}

	x := gorgonia.NewTensor(nn.g, tensor.Float32, 2, gorgonia.WithShape(len(batch), nn.stateSize), gorgonia.WithValue(tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(len(batch), nn.stateSize), tensor.WithBacking(states))))
	y := gorgonia.NewTensor(nn.g, tensor.Float32, 2, gorgonia.WithShape(len(batch), nn.actionSize), gorgonia.WithValue(tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(len(batch), nn.actionSize), tensor.WithBacking(targets))))
	l0 := gorgonia.Must(gorgonia.Mul(x, nn.w0))
	a0 := gorgonia.Must(gorgonia.Rectify(l0))
	l1 := gorgonia.Must(gorgonia.Mul(a0, nn.w1))
	loss := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(y, l1))))))

	grads, err := gorgonia.Grad(loss, nn.w0, nn.w1)
	if err != nil {
		return err
	}
	valueGrads := make([]gorgonia.ValueGrad, len(grads))
	for i, grad := range grads {
		valueGrads[i] = grad
	}

	nn.solver.Step(valueGrads)
	return nn.vmachine.RunAll()
}

// Helper functions
func float32ToFloat64(input []float32) []float64 {
	output := make([]float64, len(input))
	for i := range input {
		output[i] = float64(input[i])
	}
	return output
}

func float64ToFloat32(input []float64) []float32 {
	output := make([]float32, len(input))
	for i := range input {
		output[i] = float32(input[i])
	}
	return output
}

// Epsilon-greedy action selection
func chooseAction(nn *NeuralNetwork, state []float64, adr_datarate ttnpb.DataRateIndex) ttnpb.DataRateIndex {
	if rand.Float64() < epsilon {
		return adr_datarate
	}
	qValues, _ := nn.Predict(state)
	maxValue := qValues[0]
	bestAction := ttnpb.DataRateIndex_DATA_RATE_0
	for a := 1; a < numActions; a++ {
		if qValues[a] > maxValue {
			maxValue = qValues[a]
			bestAction = ttnpb.DataRateIndex(a)
		}
	}
	return bestAction
}

// Convert state to a float64 array
func stateToFloatArray(device Device) []float64 {
	return []float64{
		float64(device.Locationx),
		float64(device.Locationy),
		float64(device.SNR),
		float64(device.CurrentDataRate),
		device.CurrentDemodFloor,
		device.CurrentBatteryLevel,
		float64(boolToInt(device.ChannelSteering)),
	}
}

// Helper function to convert bool to int
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Environment transition function (example)
func getNextState(device Device, action ttnpb.DataRateIndex) Device {
	// Example transition, should be replaced with actual logic
	device.CurrentDataRate = action
	return device
}

func CreateSARSAModel() {
	numDevices = gatewayload
	numStates = 6 * numDevices // for each devices, 2 locations, 1 SNR, 1 current dr, 1 current demodfloor, 1 channelsteering
	// Initialize the neural network
	nn = NewNeuralNetwork(numStates, numActions)
	// Initialize experience replay memory
	memory = make([]Experience, 0, memorySize)
}

// Main function
func ScheduleSarsa(device Device, adr_datarate ttnpb.DataRateIndex) ttnpb.DataRateIndex {

	// Initialize devices

	// Online SARSA training

	state := stateToFloatArray(device)
	action := device.CurrentDataRate
	nextDevice := getNextState(device, action)
	reward := getReward(device)
	nextState := stateToFloatArray(nextDevice)
	nextAction := chooseAction(nn, nextState, adr_datarate)

	// Store experience in memory
	if len(memory) >= memorySize {
		memory = memory[1:]
	}
	memory = append(memory, Experience{State: state, Action: action, Reward: reward, NextState: nextState, NextAction: nextAction})

	// Train the neural network with a batch from memory
	if len(memory) >= batchSize {
		batch := memory[rand.Intn(len(memory)-batchSize+1) : rand.Intn(len(memory)-batchSize+1)+batchSize]
		nn.Train(batch)
	}

	fmt.Println("Prediction completed.")
	return nextAction
}

func getReward(device Device) float64 {
	return device.CurrentBatteryLevel
}
