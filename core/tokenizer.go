package core

import "github.com/pkoukk/tiktoken-go"

var _ Tokenizer = (*TikTokenizer)(nil)

const CL100KBaseName = "cl100k_base"
const CL100KBaseVocabSize = 100277 // 用于按 tokenID 直接索引 embedding

type Tokenizer interface {
	Encode(text string) ([]int, error)
	Decode(tokenIDs []int) (string, error)
	EncodingName() string
	VocabSize() int
}

type TikTokenizer struct {
	enc          *tiktoken.Tiktoken
	encodingName string
}

func NewTikTokenizer(encodingName string) (*TikTokenizer, error) {
	enc, err := tiktoken.GetEncoding(encodingName)
	if err != nil {
		return nil, err
	}

	return &TikTokenizer{
		enc:          enc,
		encodingName: encodingName,
	}, nil
}

func (t *TikTokenizer) Encode(text string) ([]int, error) {
	return t.enc.Encode(text, nil, nil), nil
}

func (t *TikTokenizer) Decode(tokenIDs []int) (string, error) {
	return t.enc.Decode(tokenIDs), nil
}

func (t *TikTokenizer) EncodingName() string {
	return t.encodingName
}

func (t *TikTokenizer) VocabSize() int {
	switch t.EncodingName() {
	case CL100KBaseName:
		return CL100KBaseVocabSize
	default:
		return -1
	}
}
