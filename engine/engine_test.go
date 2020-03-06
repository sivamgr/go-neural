package engine

import (
	"testing"

	"github.com/sivamgr/go-neural"
	"github.com/sivamgr/go-neural/persist"
	"github.com/stretchr/testify/assert"
)

func TestBasic(t *testing.T) {
	network := neural.NewNetwork(2, []int{2, 2})
	engine := New(network)
	engine.Start()

	engine.Learn([]float64{1, 2}, []float64{3, 3}, 0.1)

	out := engine.Calculate([]float64{1, 2})

	assert.Equal(t, len(out), 2)
}

func TestDump(t *testing.T) {
	network := neural.NewNetwork(3, []int{3, 3})
	engine := New(network)
	engine.Start()

	dump := persist.ToDump(network)
	dumpEng := engine.Dump()

	assert.Equal(t, dump.Enters, dumpEng.Enters)
}
