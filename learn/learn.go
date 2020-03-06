package learn

import (
	"github.com/sivamgr/go-neural"
)

type Deltas [][]float64

type Sample struct {
	In    []float64
	Ideal []float64
}

func Learn(n *neural.Network, in, ideal []float64, speed float64) {
	Backpropagation(n, in, ideal, speed)
}

func Backpropagation(n *neural.Network, in, ideal []float64, speed float64) {
	n.Calculate(in)
	last := len(n.Layers) - 1
	l := n.Layers[last]
	for i, neu := range l.Neurons {
		n.Deltas[last][i] = neu.Out * (1 - neu.Out) * (ideal[i] - neu.Out)
	}

	for i := last - 1; i >= 0; i-- {
		l := n.Layers[i]
		for j, neu := range l.Neurons {
			var sum float64 = 0
			for k, s := range neu.OutSynapses {
				sum += s.Weight * n.Deltas[i+1][k]
			}

			n.Deltas[i][j] = neu.Out * (1 - neu.Out) * sum
		}
	}

	for i, l := range n.Layers {
		for j, neu := range l.Neurons {
			for _, s := range neu.InSynapses {
				s.Weight += speed * n.Deltas[i][j] * s.In
			}
		}
	}

}
