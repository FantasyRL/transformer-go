package core

import "math"

type PositionalEncoder interface {
	// PositionInject 在向量中注入位置关系
	PositionInject(vectors [][]float64) [][]float64
	// Dim 维度
	Dim() int
}

type TextPositionalEncoder struct {
	dim int
}

func NewTextPositionalEncoder(dim int) *TextPositionalEncoder {
	return &TextPositionalEncoder{dim: dim}
}

func (t *TextPositionalEncoder) PositionInject(vectors [][]float64) [][]float64 {
	injected := make([][]float64, len(vectors))
	for pos := range vectors {
		injected[pos] = make([]float64, len(vectors[pos]))
		// 三角函数的连续性
		for j := range vectors[pos] {
			angle := float64(pos) / math.Pow(10000, float64(2*(j/2))/float64(t.dim))
			if j%2 == 0 {
				injected[pos][j] = vectors[pos][j] + math.Sin(angle)
				continue
			}
			injected[pos][j] = vectors[pos][j] + math.Cos(angle)
		}
	}
	return injected
}

func (t *TextPositionalEncoder) Dim() int {
	return t.dim
}
