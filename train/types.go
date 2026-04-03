package train

type TrainState struct {
	// RunID 是本次训练运行的唯一标识。
	RunID string `json:"run_id"`
	// CurrentEpoch 是当前已经训练到的 epoch。
	CurrentEpoch int `json:"current_epoch"`
	// GlobalStep 是累计执行的训练步数。
	GlobalStep int `json:"global_step"`
	// BestLoss 是当前已记录到的最优 loss。
	BestLoss float64 `json:"best_loss"`
	// LastLoss 是最近一次记录的 loss。
	LastLoss float64 `json:"last_loss"`
	// LastCheckpointPath 是最近一次写入的 checkpoint 路径。
	LastCheckpointPath string `json:"last_checkpoint_path"`
	// DatasetFingerprint 用于标识当前训练所使用的数据集身份。
	DatasetFingerprint string `json:"dataset_fingerprint"`
	// ConfigFingerprint 用于标识当前训练所使用的静态配置身份。
	ConfigFingerprint string `json:"config_fingerprint"`
}
