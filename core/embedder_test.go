package core

import "testing"

func TestTextEmbedderTokenIDsToVectors(t *testing.T) {
	const dim = 512
	vocabSize := CL100KBaseVocabSize

	weights := make([][]float64, vocabSize)
	for i := range weights {
		weights[i] = make([]float64, dim)
		for j := range weights[i] {
			weights[i][j] = float64(i + j)
		}
	}

	embedder := NewTextEmbedder(weights, vocabSize, dim)
	tokenIDs := []int{0, 42, vocabSize - 1}
	vectors := embedder.TokenIDsToVectors(tokenIDs)

	if len(vectors) != len(tokenIDs) {
		t.Fatalf("len(vectors) = %d, want %d", len(vectors), len(tokenIDs))
	}

	for i, tokenID := range tokenIDs {
		if len(vectors[i]) != dim {
			t.Fatalf("len(vectors[%d]) = %d, want %d", i, len(vectors[i]), dim)
		}

		if vectors[i][0] != weights[tokenID][0] {
			t.Fatalf("vectors[%d][0] = %v, want %v", i, vectors[i][0], weights[tokenID][0])
		}

		if vectors[i][dim-1] != weights[tokenID][dim-1] {
			t.Fatalf("vectors[%d][%d] = %v, want %v", i, dim-1, vectors[i][dim-1], weights[tokenID][dim-1])
		}
	}
}

func TestTextEmbedderNearestTokenID(t *testing.T) {
	weights := [][]float64{
		{0, 0, 0},
		{1, 1, 1},
		{10, 10, 10},
	}

	embedder := NewTextEmbedder(weights, len(weights), 3)

	if got := embedder.NearestTokenID([]float64{0.1, 0.2, 0.1}); got != 0 {
		t.Fatalf("NearestTokenID() = %d, want %d", got, 0)
	}

	if got := embedder.NearestTokenID([]float64{0.9, 1.2, 1.1}); got != 1 {
		t.Fatalf("NearestTokenID() = %d, want %d", got, 1)
	}

	if got := embedder.NearestTokenID([]float64{9.5, 10.2, 9.8}); got != 2 {
		t.Fatalf("NearestTokenID() = %d, want %d", got, 2)
	}
}
