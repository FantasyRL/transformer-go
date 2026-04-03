package config

type Config struct {
	// Experiment 是本次实验的标识信息。
	Experiment ExperimentConfig `yaml:"experiment" json:"experiment"`
	// Dataset 描述训练、验证、测试数据集的来源。
	Dataset DatasetConfig `yaml:"dataset" json:"dataset"`
	// Model 描述模型结构相关的静态超参数。
	Model ModelConfig `yaml:"model" json:"model"`
	// Training 描述训练过程中的静态超参数。
	Training TrainingConfig `yaml:"training" json:"training"`
	// Output 描述本次训练产物的输出位置。
	Output OutputConfig `yaml:"output" json:"output"`
}

type ExperimentConfig struct {
	// Name 是本次实验的人类可读名称。
	Name string `yaml:"name" json:"name"`
}

type DatasetConfig struct {
	// TrainPath 是训练集文件路径。
	TrainPath string `yaml:"train_path" json:"train_path"`
	// ValidPath 是验证集文件路径。
	ValidPath string `yaml:"valid_path" json:"valid_path"`
	// TestPath 是测试集文件路径。
	TestPath string `yaml:"test_path" json:"test_path"`
}

type ModelConfig struct {
	// VocabSize 是词表大小。
	VocabSize int `yaml:"vocab_size" json:"vocab_size"`
	// DModel 是 token embedding 和隐藏状态的维度。
	DModel int `yaml:"d_model" json:"d_model"`
	// DFF 是前馈网络中间层的维度。
	DFF int `yaml:"d_ff" json:"d_ff"`
	// Heads 是 multi-head attention 的头数。
	Heads int `yaml:"heads" json:"heads"`
	// NLayers 是 encoder 或 decoder 堆叠的层数。
	NLayers int `yaml:"n_layers" json:"n_layers"`
	// MaxSeqLen 是模型支持的最大序列长度。
	MaxSeqLen int `yaml:"max_seq_len" json:"max_seq_len"`
}

type TrainingConfig struct {
	// BatchSize 是每次训练迭代处理的样本数。
	BatchSize int `yaml:"batch_size" json:"batch_size"`
	// LearningRate 是优化器的学习率。
	LearningRate float64 `yaml:"learning_rate" json:"learning_rate"`
	// Epochs 是完整遍历训练集的轮数。
	Epochs int `yaml:"epochs" json:"epochs"`
	// Seed 是用于保证可复现性的随机种子。
	Seed int64 `yaml:"seed" json:"seed"`
}

type OutputConfig struct {
	// RunDir 是本次训练运行目录的根路径。
	RunDir string `yaml:"run_dir" json:"run_dir"`
	// CheckpointDir 是 checkpoint 保存目录。
	CheckpointDir string `yaml:"checkpoint_dir" json:"checkpoint_dir"`
	// ModelPath 是最终模型参数输出路径。
	ModelPath string `yaml:"model_path" json:"model_path"`
	// LogDir 是训练日志输出目录。
	LogDir string `yaml:"log_dir" json:"log_dir"`
}

type EmbedderConfig struct {
	// VocabSize 是 embedding 矩阵覆盖的 token 数量。
	VocabSize int `yaml:"vocab_size" json:"vocab_size"`
	// Dim 是每个 token embedding 的维度。
	Dim int `yaml:"dim" json:"dim"`
	// Weights 是 embedding 矩阵参数。
	Weights [][]float64 `yaml:"weights" json:"weights"`
}

type LinearConfig struct {
	// InputDim 是线性层输入向量维度。
	InputDim int `yaml:"input_dim" json:"input_dim"`
	// OutputDim 是线性层输出向量维度。
	OutputDim int `yaml:"output_dim" json:"output_dim"`
	// Weights 是线性层权重矩阵。
	Weights [][]float64 `yaml:"weights" json:"weights"`
	// Bias 是线性层偏置向量。
	Bias []float64 `yaml:"bias" json:"bias"`
}

type MultiHeadAttentionConfig struct {
	// QLinear 是 query 投影层参数。
	QLinear LinearConfig `yaml:"q_linear" json:"q_linear"`
	// KLinear 是 key 投影层参数。
	KLinear LinearConfig `yaml:"k_linear" json:"k_linear"`
	// VLinear 是 value 投影层参数。
	VLinear LinearConfig `yaml:"v_linear" json:"v_linear"`
	// OutLinear 是 attention 输出投影层参数。
	OutLinear LinearConfig `yaml:"out_linear" json:"out_linear"`
	// Heads 是 attention 头数。
	Heads int `yaml:"heads" json:"heads"`
}

type LayerNormConfig struct {
	// Gamma 是 layer norm 的缩放参数。
	Gamma []float64 `yaml:"gamma" json:"gamma"`
	// Beta 是 layer norm 的平移参数。
	Beta []float64 `yaml:"beta" json:"beta"`
	// Eps 是防止除零的数值稳定项。
	Eps float64 `yaml:"eps" json:"eps"`
}

type FeedForwardConfig struct {
	// Linear1 是第一层线性变换参数。
	Linear1 LinearConfig `yaml:"linear1" json:"linear1"`
	// Linear2 是第二层线性变换参数。
	Linear2 LinearConfig `yaml:"linear2" json:"linear2"`
}

type CheckpointMeta struct {
	// RunID 是生成该 checkpoint 的运行标识。
	RunID string `json:"run_id"`
	// Epoch 是该 checkpoint 对应的 epoch。
	Epoch int `json:"epoch"`
	// Step 是该 checkpoint 对应的全局步数。
	Step int `json:"step"`
	// DatasetFingerprint 用于校验 checkpoint 是否来自同一数据集。
	DatasetFingerprint string `json:"dataset_fingerprint"`
	// ConfigFingerprint 用于校验 checkpoint 是否来自同一静态配置。
	ConfigFingerprint string `json:"config_fingerprint"`
}
