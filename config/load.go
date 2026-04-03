package config

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"
)

func Load(path string) (*Config, error) {
	v := viper.New()
	v.SetConfigFile(path)

	ext := strings.TrimPrefix(filepath.Ext(path), ".")
	if ext != "" {
		v.SetConfigType(ext)
	}

	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshal config: %w", err)
	}

	return &cfg, nil
}
