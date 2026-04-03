package train

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/FantasyRL/transformer-go/config"
)

func LoadTrainState(path string) (*TrainState, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read train state: %w", err)
	}

	var state TrainState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("unmarshal train state: %w", err)
	}

	return &state, nil
}

func SaveTrainState(path string, state *TrainState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create train state dir: %w", err)
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal train state: %w", err)
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write train state: %w", err)
	}

	return nil
}

func LoadCheckpointMeta(path string) (*config.CheckpointMeta, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read checkpoint meta: %w", err)
	}

	var meta config.CheckpointMeta
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, fmt.Errorf("unmarshal checkpoint meta: %w", err)
	}

	return &meta, nil
}

func SaveCheckpointMeta(path string, meta *config.CheckpointMeta) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create checkpoint meta dir: %w", err)
	}

	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal checkpoint meta: %w", err)
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write checkpoint meta: %w", err)
	}

	return nil
}
