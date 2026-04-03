package core

var _ Embedder = (*TextEmbedder)(nil)

type Embedder interface {
	// TokenIDsToVectors 将tokenID转换为向量
	TokenIDsToVectors(tokenIDs []int) [][]float64
	// NearestTokenID 返回与输入向量最接近的tokenID
	NearestTokenID(vector []float64) int
	// VocabSize 可接受的tokenID映射范围
	VocabSize() int
	// Dim 维度
	Dim() int
}

type TextEmbedder struct {
	weights   [][]float64
	vocabSize int
	dim       int
}

func NewTextEmbedder(
	weights [][]float64,
	vocabSize int,
	dim int,
) *TextEmbedder {
	return &TextEmbedder{
		weights:   weights,
		vocabSize: vocabSize,
		dim:       dim,
	}
}

func (t TextEmbedder) TokenIDsToVectors(tokenIDs []int) [][]float64 {
	vectors := make([][]float64, len(tokenIDs))
	for i, tokenID := range tokenIDs {
		vectors[i] = t.weights[tokenID]
	}
	return vectors
}

func (t TextEmbedder) VocabSize() int {
	return t.vocabSize
}

func (t TextEmbedder) Dim() int {
	return t.dim
}

func (t TextEmbedder) NearestTokenID(vector []float64) int {
	bestTokenID := -1
	bestDistance := 0.0

	for tokenID, weight := range t.weights {
		distance := squaredEuclideanDistance(vector, weight)
		if bestTokenID == -1 || distance < bestDistance {
			bestTokenID = tokenID
			bestDistance = distance
		}
	}

	return bestTokenID
}

func squaredEuclideanDistance(a, b []float64) float64 {
	distance := 0.0
	for i := range a {
		diff := a[i] - b[i]
		distance += diff * diff
	}
	return distance
}
