package core

import (
	"math"
	"testing"
)

func TestTextPositionalEncoderAddsToExistingVectors(t *testing.T) {
	encoder := NewTextPositionalEncoder(4)
	vectors := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
	}

	injected := encoder.PositionInject(vectors)

	assertFloatAlmostEqual(t, injected[0][0], 1)
	assertFloatAlmostEqual(t, injected[0][1], 3)
	assertFloatAlmostEqual(t, injected[0][2], 3)
	assertFloatAlmostEqual(t, injected[0][3], 5)

	assertFloatAlmostEqual(t, injected[1][0], 5+math.Sin(1))
	assertFloatAlmostEqual(t, injected[1][1], 6+math.Cos(1))
	assertFloatAlmostEqual(t, injected[1][2], 7+math.Sin(0.01))
	assertFloatAlmostEqual(t, injected[1][3], 8+math.Cos(0.01))

}

func assertFloatAlmostEqual(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("got %v, want %v", got, want)
	}
}
