package core

import "testing"

func TestTikTokenizerEncodeDecodeRoundTrip(t *testing.T) {
	tokenizer, err := NewTikTokenizer(CL100KBaseName)
	if err != nil {
		t.Fatalf("NewTikTokenizer returned error: %v", err)
	}

	text := "Hello, Transformer!"
	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		t.Fatalf("Encode returned error: %v", err)
	}

	if len(tokenIDs) == 0 {
		t.Fatal("Encode returned empty token IDs")
	}

	decoded, err := tokenizer.Decode(tokenIDs)
	if err != nil {
		t.Fatalf("Decode returned error: %v", err)
	}

	if decoded != text {
		t.Fatalf("Decode(Encode(%q)) = %q, want %q", text, decoded, text)
	}
}

func TestTikTokenizerEncodeEmptyString(t *testing.T) {
	tokenizer, err := NewTikTokenizer("cl100k_base")
	if err != nil {
		t.Fatalf("NewTikTokenizer returned error: %v", err)
	}

	tokenIDs, err := tokenizer.Encode("")
	if err != nil {
		t.Fatalf("Encode returned error: %v", err)
	}

	if len(tokenIDs) != 0 {
		t.Fatalf("Encode(\"\") returned %d token IDs, want 0", len(tokenIDs))
	}
}
