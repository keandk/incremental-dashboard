// Global variables to store data and charts
let forwardData = null;
let forgettingData = null;
let hoverMode = 'forward'; // 'forward' | 'forgetting'
let currentModel = 'replay-30';
let charts = {};

// Dataset (Gantt) globals
let partitionsManifest = null;
let datasetChart = null;

// Training configuration data
const trainingConfigs = {
  'replay-30': {
    // Basic settings
    use_exemplars: '30%',
    loss_type: 'rldam',
    rldam_C: 1.0,
    rldam_nth_power: 0.25,
    model_name: 'microsoft/graphcodebert-base',
    num_epochs: 15,
    learning_rate: 2e-5,
    ewc_lambda: null,

    // Model architecture
    tokenizer_path: '/kaggle/input/tokenizer-gcb-dacn/scratch-combined-tokenizer',
    max_length: 512,
    max_chunks: 4,
    dropout_rate: 0.1,
    chunk_aggregation_method: 'attention',

    // Training hyperparameters
    per_device_train_batch_size: 32,
    per_device_eval_batch_size: 32,
    gradient_accumulation_steps: 8,
    warmup_ratio: 0.1,
    weight_decay: 0.01,

    // Training settings
    early_stopping_patience: 5,
    save_strategy: 'epoch',
    evaluation_strategy: 'epoch',
    logging_strategy: 'epoch',
    logging_steps: 1,
    save_total_limit: 2,
    load_best_model_at_end: true,
    metric_for_best_model: 'f1_macro',
    greater_is_better: true,

    // Memory optimization
    gradient_checkpointing: true,
    fp16: true,
    auto_find_batch_size: true,
    dataloader_num_workers: 4,
    dataloader_pin_memory: true,
    remove_unused_columns: false,

    // Exemplar/Replay settings
    exemplar_selection_strategy: 'wru',
    exemplar_combine_strategy: 'concat',
    exemplar_unmask_strategy: 'full',
    exemplar_unmask_ratio: 1.0,

    // Task settings
    partition_manifest_path: '/kaggle/input/ewc-partitions/temporal_partitions_w_adaptation/partitions_manifest.json',
    dataset_name: 'norm-source',
    num_tasks: 9,
    output_base_dir: './models/incremental-graphcodebert-improved',

    // Resume settings
    resume_from_task: null,
    resume_from_checkpoint: null,
    model_path: null,
    wandb_run_id: null,
    wandb_run_ids: null,

    key_differences: ['use_exemplars: 30%', 'num_epochs: 15']
  },
  'replay-60': {
    // Basic settings
    use_exemplars: '60%',
    loss_type: 'rldam',
    rldam_C: 1.0,
    rldam_nth_power: 0.25,
    model_name: 'microsoft/graphcodebert-base',
    num_epochs: 15,
    learning_rate: 2e-5,
    ewc_lambda: null,

    // Model architecture
    tokenizer_path: '/kaggle/input/tokenizer-gcb-dacn/scratch-combined-tokenizer',
    max_length: 512,
    max_chunks: 4,
    dropout_rate: 0.1,
    chunk_aggregation_method: 'attention',

    // Training hyperparameters
    per_device_train_batch_size: 32,
    per_device_eval_batch_size: 32,
    gradient_accumulation_steps: 8,
    warmup_ratio: 0.1,
    weight_decay: 0.01,

    // Training settings
    early_stopping_patience: 5,
    save_strategy: 'epoch',
    evaluation_strategy: 'epoch',
    logging_strategy: 'epoch',
    logging_steps: 1,
    save_total_limit: 2,
    load_best_model_at_end: true,
    metric_for_best_model: 'f1_macro',
    greater_is_better: true,

    // Memory optimization
    gradient_checkpointing: true,
    fp16: true,
    auto_find_batch_size: true,
    dataloader_num_workers: 4,
    dataloader_pin_memory: true,
    remove_unused_columns: false,

    // Exemplar/Replay settings
    exemplar_selection_strategy: 'wru',
    exemplar_combine_strategy: 'concat',
    exemplar_unmask_strategy: 'full',
    exemplar_unmask_ratio: 1.0,

    // Task settings
    partition_manifest_path: '/kaggle/input/ewc-partitions/temporal_partitions_w_adaptation/partitions_manifest.json',
    dataset_name: 'norm-source',
    num_tasks: 9,
    output_base_dir: './models/incremental-graphcodebert-improved',

    // Resume settings
    resume_from_task: null,
    resume_from_checkpoint: null,
    model_path: null,
    wandb_run_id: null,
    wandb_run_ids: null,

    key_differences: ['use_exemplars: 60%', 'num_epochs: 15']
  },
  'ewc-2000': {
    // Basic settings
    use_exemplars: 'none',
    loss_type: 'rldam',
    rldam_C: 1.0,
    rldam_nth_power: 0.25,
    model_name: 'microsoft/graphcodebert-base',
    num_epochs: 15,
    learning_rate: 2e-5,
    ewc_lambda: 2000,

    // Model architecture
    tokenizer_path: '/kaggle/input/tokenizer-gcb-dacn/scratch-combined-tokenizer',
    max_length: 512,
    max_chunks: 4,
    dropout_rate: 0.1,
    chunk_aggregation_method: 'attention',

    // Training hyperparameters
    per_device_train_batch_size: 32,
    per_device_eval_batch_size: 32,
    gradient_accumulation_steps: 8,
    warmup_ratio: 0.1,
    weight_decay: 0.01,

    // Training settings
    early_stopping_patience: 5,
    save_strategy: 'epoch',
    evaluation_strategy: 'epoch',
    logging_strategy: 'epoch',
    logging_steps: 1,
    save_total_limit: 2,
    load_best_model_at_end: true,
    metric_for_best_model: 'f1_macro',
    greater_is_better: true,

    // Memory optimization
    gradient_checkpointing: true,
    fp16: true,
    auto_find_batch_size: true,
    dataloader_num_workers: 4,
    dataloader_pin_memory: true,
    remove_unused_columns: false,

    // EWC settings
    ewc_lambda: 2000,

    // Task settings
    partition_manifest_path: '/kaggle/input/ewc-partitions/temporal_partitions_w_adaptation/partitions_manifest.json',
    dataset_name: 'norm-source',
    num_tasks: 9,
    output_base_dir: './models/incremental-graphcodebert-improved',

    // Resume settings
    resume_from_task: null,
    resume_from_checkpoint: null,
    model_path: null,
    wandb_run_id: null,
    wandb_run_ids: null,

    adaptive_training: true,
    key_differences: ['use_exemplars: none', 'ewc_lambda: 2000', 'num_epochs: 15', 'adaptive_training: true'],
    adaptive_methods: {
      patience: `if task_size < 100: return 2
elif task_size < 500: return 3
elif task_size < 1000: return 4
else: return 5`,
      learning_rate: `if task_size < 100: return base_lr * 3.0
elif task_size < 400: return base_lr * 2.0
else: return base_lr`,
      warmup: `if task_size < 100: return 0.0
elif task_size < 400: return 0.05
else: return 0.1`,
      batch_size: `if task_size < 300: return 8
elif task_size < 400: return 16
else: return 32`,
      ewc_lambda: `if task_size < 100: return base_lambda * 0.5
elif task_size < 400: return base_lambda * 0.75
else: return base_lambda`,
      min_epochs: `if task_size < 100: return 5
elif task_size < 400: return 3
else: return 1`
    },
    adaptive_rules: [
      'Tasks < 100 samples: EWC λ=1000 (50%), LR=6e-5 (3x), patience=2, warmup=0.0, min_epochs=5',
      'Tasks 100-400 samples: EWC λ=1500 (75%), LR=4e-5 (2x), patience=3, warmup=0.05, min_epochs=3',
      'Tasks 400-1000 samples: EWC λ=2000 (100%), LR=2e-5 (1x), patience=4, warmup=0.1, min_epochs=1',
      'Tasks > 1000 samples: EWC λ=2000 (100%), LR=2e-5 (1x), patience=5, warmup=0.1, min_epochs=1'
    ]
  }
};

// Column metric keys for the main metrics table (order must match table headers)
const METRIC_COLUMN_KEYS = [
  'precision_macro',
  'precision_micro',
  'precision_samples',
  'recall_macro',
  'recall_micro',
  'recall_samples',
  'f1_macro',
  'f1_micro',
  'f1_samples',
  'roc_auc_macro',
  'pr_auc_macro'
];

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
  loadData();
});

// Load JSON data
async function loadData() {
  try {
    const [forwardResp, forgettingResp] = await Promise.all([
      fetch('./forward_evaluation_summary.json'),
      fetch('./forgetting_evaluation_summary.json').catch(() => null)
    ]);
    forwardData = await forwardResp.json();
    if (forgettingResp && forgettingResp.ok) {
      try {
        forgettingData = await forgettingResp.json();
      } catch (_) {
        forgettingData = null;
      }
    }
    console.log('Data loaded successfully:', Object.keys(forwardData));
    initializeCharts();
    // If per-class tab is active on load, populate its table
    const perClassTab = document.getElementById('per-class');
    if (perClassTab && perClassTab.classList.contains('active')) {
      populatePerClassF1Table();
    }
  } catch (error) {
    console.error('Error loading data:', error);
    // For demo purposes, create mock data
    createMockData();
  } finally {
    // Load dataset partitions manifest regardless of forward metrics load outcome
    loadPartitionsManifest();
  }
}

// Load partitions manifest for dataset (Gantt)
async function loadPartitionsManifest() {
  try {
    const resp = await fetch('./partitions_manifest.json');
    partitionsManifest = await resp.json();
    updateDatasetStats(partitionsManifest);
    // If dataset tab is active, render immediately
    const dsTab = document.getElementById('dataset');
    if (dsTab && dsTab.classList.contains('active')) {
      renderDatasetGantt();
    }
  } catch (e) {
    console.error('Error loading partitions manifest:', e);
  }
}

function ensureDatasetInitialized() {
  if (!partitionsManifest) return;
  if (!datasetChart) {
    renderDatasetGantt();
  }
}

function parseTime(value) {
  if (!value) return null;
  try {
    // Normalize to ISO-like format and trim fractional seconds to milliseconds
    const withT = value.replace(' ', 'T');
    const trimmed = withT.replace(/\.(\d{3})\d+$/, '.$1');
    const d = new Date(trimmed);
    if (!isNaN(d.getTime())) return d;
    const d2 = new Date(value);
    return isNaN(d2.getTime()) ? null : d2;
  } catch (_) {
    return null;
  }
}

function formatDateShort(d) {
  if (!d) return 'N/A';
  try {
    return d.toISOString().slice(0, 10);
  } catch (_) {
    return 'N/A';
  }
}

function computeGlobalTimeSpan(parts) {
  let minStart = null;
  let maxEnd = null;
  parts.forEach(p => {
    const a = p.time_coverage_adapt;
    const f = p.time_coverage_eval_ft;
    const b = p.time_coverage_eval_bwt;
    const starts = [a?.start, f?.start, b?.start].map(parseTime).filter(Boolean);
    const ends = [a?.end, f?.end, b?.end].map(parseTime).filter(Boolean);
    starts.forEach(s => { if (!minStart || s < minStart) minStart = s; });
    ends.forEach(e => { if (!maxEnd || e > maxEnd) maxEnd = e; });
  });
  return { minStart, maxEnd };
}

function updateDatasetStats(manifest) {
  if (!manifest) return;
  const nameEl = document.getElementById('ds-name');
  const cEl = document.getElementById('ds-contracts');
  const pEl = document.getElementById('ds-partitions');
  const spanEl = document.getElementById('ds-timespan');
  if (!nameEl || !cEl || !pEl || !spanEl) return;

  nameEl.textContent = manifest.dataset_name || '—';
  cEl.textContent = (manifest.total_contracts ?? '—').toString();
  const partitions = Array.isArray(manifest.partitions) ? manifest.partitions : [];
  pEl.textContent = String(partitions.length);
  const { minStart, maxEnd } = computeGlobalTimeSpan(partitions);
  spanEl.textContent = `${formatDateShort(minStart)} → ${formatDateShort(maxEnd)}`;
}

function buildGanttDatasets(partitions) {
  const adaptColor = 'rgba(59, 130, 246, 0.7)'; // blue
  const ftColor = 'rgba(16, 185, 129, 0.7)'; // green
  const bwtColor = 'rgba(239, 68, 68, 0.7)'; // red

  const adapt = [];
  const ft = [];
  const bwt = [];

  partitions.forEach(p => {
    const y = `P${p.partition_id}`;
    if (p.time_coverage_adapt) {
      const start = parseTime(p.time_coverage_adapt.start);
      const end = parseTime(p.time_coverage_adapt.end);
      if (start && end) {
        adapt.push({ x: [start, end], y, meta: { rows: p.time_coverage_adapt.rows, type: 'Adapt', vulnCounts: p.adapt_label_counts } });
      }
    }
    if (p.time_coverage_eval_ft) {
      const start = parseTime(p.time_coverage_eval_ft.start);
      const end = parseTime(p.time_coverage_eval_ft.end);
      if (start && end) {
        ft.push({ x: [start, end], y, meta: { rows: p.time_coverage_eval_ft.rows, type: 'Eval FT', vulnCounts: p.eval_ft_label_counts } });
      }
    }
    if (p.time_coverage_eval_bwt) {
      const start = parseTime(p.time_coverage_eval_bwt.start);
      const end = parseTime(p.time_coverage_eval_bwt.end);
      if (start && end) {
        bwt.push({ x: [start, end], y, meta: { rows: p.time_coverage_eval_bwt.rows, type: 'Eval BWT', vulnCounts: p.eval_bwt_label_counts } });
      }
    }
  });

  return [
    {
      label: 'Adapt',
      data: adapt,
      backgroundColor: adaptColor,
      borderColor: adaptColor.replace('0.7', '1'),
      borderWidth: 1,
      borderSkipped: false,
      barPercentage: 0.9,
      borderRadius: 6,
    },
    {
      label: 'Eval FT',
      data: ft,
      backgroundColor: ftColor,
      borderColor: ftColor.replace('0.7', '1'),
      borderWidth: 1,
      borderSkipped: false,
      barPercentage: 0.9,
      borderRadius: 6,
    },
    {
      label: 'Eval BWT',
      data: bwt,
      backgroundColor: bwtColor,
      borderColor: bwtColor.replace('0.7', '1'),
      borderWidth: 1,
      borderSkipped: false,
      barPercentage: 0.9,
      borderRadius: 6,
    },
  ];
}

function renderDatasetGantt() {
  const manifest = partitionsManifest;
  if (!manifest || !manifest.partitions) return;
  const partitions = manifest.partitions;

  const ctx = document.getElementById('datasetChart');
  if (!ctx) return;

  const datasets = buildGanttDatasets(partitions);
  const { minStart, maxEnd } = computeGlobalTimeSpan(partitions);

  if (datasetChart) {
    datasetChart.destroy();
  }

  datasetChart = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: { xAxisKey: 'x', yAxisKey: 'y' },
      indexAxis: 'y',
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          mode: 'nearest',
          intersect: false,
          position: 'nearest',
          callbacks: {
            title: (items) => {
              if (!items || !items.length) return '';
              const y = items[0].raw?.y;
              return `Partition ${y.replace('P', '')}`;
            },
            beforeLabel: (ctx) => {
              const raw = ctx.raw || {};
              const range = raw.x || [];
              const start = formatDateShort(range[0]);
              const end = formatDateShort(range[1]);
              const rows = raw.meta?.rows;
              const seg = ctx.dataset?.label || '';
              const rowsText = typeof rows === 'number' ? `, ${rows} rows` : '';
              return `${seg}: ${start} → ${end}${rowsText}`;
            },
            label: (ctx) => {
              const vulnText = formatVulnCountsForTooltip(ctx.raw.meta?.vulnCounts);
              // Return array of strings to ensure proper line breaks
              return vulnText ? vulnText.split('\n') : ['No vulnerabilities'];
            }
          },
          displayColors: true,
          backgroundColor: 'rgba(0, 0, 0, 0.95)',
          titleColor: '#fff',
          bodyColor: '#fff',
          borderColor: '#3b82f6',
          borderWidth: 2,
          cornerRadius: 8,
          padding: { top: 16, right: 20, bottom: 16, left: 20 },
          boxWidth: 12,
          boxHeight: 12,
          usePointStyle: false,
          titleFont: { size: 14, weight: 'bold' },
          bodyFont: { size: 12 },
          titleMarginBottom: 10,
          bodySpacing: 6,
          multiKeyBackground: 'transparent'
        },
        title: { display: false }
      },
      scales: {
        x: {
          type: 'time',
          time: { unit: 'month' },
          min: minStart ? minStart.getTime() : undefined,
          max: maxEnd ? maxEnd.getTime() : undefined,
          title: { display: true, text: 'Date' }
        },
        y: {
          type: 'category',
          title: { display: true, text: 'Partitions' }
        }
      }
    }
  });
}

// Create mock data for testing
function createMockData() {
  forwardData = {
    'replay-30': {
      'per_task': {
        '1': {
          'baseline': {
            'precision_macro': 0.55, 'precision_micro': 0.39, 'precision_samples': 0.18,
            'recall_macro': 0.59, 'recall_micro': 0.39, 'recall_samples': 0.76,
            'f1_macro': 0.38, 'f1_micro': 0.39, 'f1_samples': 0.28,
            'roc_auc_macro': 0.64, 'pr_auc_macro': 0.28
          }
        },
        '2': {
          'baseline': {
            'precision_macro': 0.62, 'precision_micro': 0.45, 'precision_samples': 0.22,
            'recall_macro': 0.64, 'recall_micro': 0.45, 'recall_samples': 0.78,
            'f1_macro': 0.43, 'f1_micro': 0.45, 'f1_samples': 0.32,
            'roc_auc_macro': 0.68, 'pr_auc_macro': 0.31
          }
        },
        '3': {
          'baseline': {
            'precision_macro': 0.58, 'precision_micro': 0.42, 'precision_samples': 0.20,
            'recall_macro': 0.61, 'recall_micro': 0.42, 'recall_samples': 0.75,
            'f1_macro': 0.41, 'f1_micro': 0.42, 'f1_samples': 0.30,
            'roc_auc_macro': 0.66, 'pr_auc_macro': 0.29
          }
        },
        '4': {
          'baseline': {
            'precision_macro': 0.65, 'precision_micro': 0.48, 'precision_samples': 0.24,
            'recall_macro': 0.67, 'recall_micro': 0.48, 'recall_samples': 0.80,
            'f1_macro': 0.47, 'f1_micro': 0.48, 'f1_samples': 0.35,
            'roc_auc_macro': 0.71, 'pr_auc_macro': 0.33
          }
        },
        '5': {
          'baseline': {
            'precision_macro': 0.61, 'precision_micro': 0.44, 'precision_samples': 0.21,
            'recall_macro': 0.63, 'recall_micro': 0.44, 'recall_samples': 0.77,
            'f1_macro': 0.44, 'f1_micro': 0.44, 'f1_samples': 0.32,
            'roc_auc_macro': 0.69, 'pr_auc_macro': 0.30
          }
        },
        '6': {
          'baseline': {
            'precision_macro': 0.63, 'precision_micro': 0.46, 'precision_samples': 0.23,
            'recall_macro': 0.65, 'recall_micro': 0.46, 'recall_samples': 0.79,
            'f1_macro': 0.45, 'f1_micro': 0.46, 'f1_samples': 0.33,
            'roc_auc_macro': 0.70, 'pr_auc_macro': 0.32
          }
        },
        '7': {
          'baseline': {
            'precision_macro': 0.59, 'precision_micro': 0.43, 'precision_samples': 0.20,
            'recall_macro': 0.62, 'recall_micro': 0.43, 'recall_samples': 0.76,
            'f1_macro': 0.42, 'f1_micro': 0.43, 'f1_samples': 0.31,
            'roc_auc_macro': 0.67, 'pr_auc_macro': 0.29
          }
        },
        '8': {
          'baseline': {
            'precision_macro': 0.64, 'precision_micro': 0.47, 'precision_samples': 0.24,
            'recall_macro': 0.66, 'recall_micro': 0.47, 'recall_samples': 0.80,
            'f1_macro': 0.46, 'f1_micro': 0.47, 'f1_samples': 0.34,
            'roc_auc_macro': 0.72, 'pr_auc_macro': 0.34
          }
        },
        '9': {
          'baseline': {
            'precision_macro': 0.60, 'precision_micro': 0.44, 'precision_samples': 0.22,
            'recall_macro': 0.63, 'recall_micro': 0.44, 'recall_samples': 0.78,
            'f1_macro': 0.43, 'f1_micro': 0.44, 'f1_samples': 0.32,
            'roc_auc_macro': 0.68, 'pr_auc_macro': 0.31
          }
        }
      }
    },
    'replay-60': {
      'per_task': {
        '1': {
          'baseline': {
            'precision_macro': 0.57, 'precision_micro': 0.41, 'precision_samples': 0.19,
            'recall_macro': 0.61, 'recall_micro': 0.41, 'recall_samples': 0.77,
            'f1_macro': 0.40, 'f1_micro': 0.41, 'f1_samples': 0.29,
            'roc_auc_macro': 0.65, 'pr_auc_macro': 0.29
          }
        },
        '2': {
          'baseline': {
            'precision_macro': 0.64, 'precision_micro': 0.47, 'precision_samples': 0.23,
            'recall_macro': 0.66, 'recall_micro': 0.47, 'recall_samples': 0.79,
            'f1_macro': 0.45, 'f1_micro': 0.47, 'f1_samples': 0.34,
            'roc_auc_macro': 0.70, 'pr_auc_macro': 0.32
          }
        },
        '3': {
          'baseline': {
            'precision_macro': 0.60, 'precision_micro': 0.44, 'precision_samples': 0.21,
            'recall_macro': 0.63, 'recall_micro': 0.44, 'recall_samples': 0.76,
            'f1_macro': 0.43, 'f1_micro': 0.44, 'f1_samples': 0.31,
            'roc_auc_macro': 0.68, 'pr_auc_macro': 0.30
          }
        },
        '4': {
          'baseline': {
            'precision_macro': 0.67, 'precision_micro': 0.50, 'precision_samples': 0.25,
            'recall_macro': 0.69, 'recall_micro': 0.50, 'recall_samples': 0.81,
            'f1_macro': 0.49, 'f1_micro': 0.50, 'f1_samples': 0.36,
            'roc_auc_macro': 0.73, 'pr_auc_macro': 0.35
          }
        },
        '5': {
          'baseline': {
            'precision_macro': 0.63, 'precision_micro': 0.46, 'precision_samples': 0.22,
            'recall_macro': 0.65, 'recall_micro': 0.46, 'recall_samples': 0.78,
            'f1_macro': 0.46, 'f1_micro': 0.46, 'f1_samples': 0.33,
            'roc_auc_macro': 0.71, 'pr_auc_macro': 0.32
          }
        },
        '6': {
          'baseline': {
            'precision_macro': 0.65, 'precision_micro': 0.48, 'precision_samples': 0.24,
            'recall_macro': 0.67, 'recall_micro': 0.48, 'recall_samples': 0.80,
            'f1_macro': 0.47, 'f1_micro': 0.48, 'f1_samples': 0.35,
            'roc_auc_macro': 0.72, 'pr_auc_macro': 0.34
          }
        },
        '7': {
          'baseline': {
            'precision_macro': 0.61, 'precision_micro': 0.45, 'precision_samples': 0.21,
            'recall_macro': 0.64, 'recall_micro': 0.45, 'recall_samples': 0.77,
            'f1_macro': 0.44, 'f1_micro': 0.45, 'f1_samples': 0.32,
            'roc_auc_macro': 0.69, 'pr_auc_macro': 0.31
          }
        },
        '8': {
          'baseline': {
            'precision_macro': 0.66, 'precision_micro': 0.49, 'precision_samples': 0.25,
            'recall_macro': 0.68, 'recall_micro': 0.49, 'recall_samples': 0.81,
            'f1_macro': 0.48, 'f1_micro': 0.49, 'f1_samples': 0.36,
            'roc_auc_macro': 0.74, 'pr_auc_macro': 0.36
          }
        },
        '9': {
          'baseline': {
            'precision_macro': 0.62, 'precision_micro': 0.46, 'precision_samples': 0.23,
            'recall_macro': 0.65, 'recall_micro': 0.46, 'recall_samples': 0.79,
            'f1_macro': 0.45, 'f1_micro': 0.46, 'f1_samples': 0.34,
            'roc_auc_macro': 0.70, 'pr_auc_macro': 0.33
          }
        }
      }
    },
    'ewc-2000': {
      'per_task': {
        '1': {
          'baseline': {
            'precision_macro': 0.52, 'precision_micro': 0.36, 'precision_samples': 0.16,
            'recall_macro': 0.56, 'recall_micro': 0.36, 'recall_samples': 0.74,
            'f1_macro': 0.35, 'f1_micro': 0.36, 'f1_samples': 0.26,
            'roc_auc_macro': 0.62, 'pr_auc_macro': 0.26
          }
        },
        '2': {
          'baseline': {
            'precision_macro': 0.59, 'precision_micro': 0.42, 'precision_samples': 0.20,
            'recall_macro': 0.61, 'recall_micro': 0.42, 'recall_samples': 0.76,
            'f1_macro': 0.41, 'f1_micro': 0.42, 'f1_samples': 0.30,
            'roc_auc_macro': 0.66, 'pr_auc_macro': 0.29
          }
        },
        '3': {
          'baseline': {
            'precision_macro': 0.55, 'precision_micro': 0.39, 'precision_samples': 0.18,
            'recall_macro': 0.58, 'recall_micro': 0.39, 'recall_samples': 0.73,
            'f1_macro': 0.38, 'f1_micro': 0.39, 'f1_samples': 0.28,
            'roc_auc_macro': 0.64, 'pr_auc_macro': 0.27
          }
        },
        '4': {
          'baseline': {
            'precision_macro': 0.61, 'precision_micro': 0.44, 'precision_samples': 0.22,
            'recall_macro': 0.63, 'recall_micro': 0.44, 'recall_samples': 0.77,
            'f1_macro': 0.44, 'f1_micro': 0.44, 'f1_samples': 0.32,
            'roc_auc_macro': 0.68, 'pr_auc_macro': 0.31
          }
        },
        '5': {
          'baseline': {
            'precision_macro': 0.58, 'precision_micro': 0.41, 'precision_samples': 0.19,
            'recall_macro': 0.60, 'recall_micro': 0.41, 'recall_samples': 0.75,
            'f1_macro': 0.41, 'f1_micro': 0.41, 'f1_samples': 0.30,
            'roc_auc_macro': 0.66, 'pr_auc_macro': 0.28
          }
        },
        '6': {
          'baseline': {
            'precision_macro': 0.60, 'precision_micro': 0.43, 'precision_samples': 0.21,
            'recall_macro': 0.62, 'recall_micro': 0.43, 'recall_samples': 0.76,
            'f1_macro': 0.43, 'f1_micro': 0.43, 'f1_samples': 0.31,
            'roc_auc_macro': 0.67, 'pr_auc_macro': 0.30
          }
        },
        '7': {
          'baseline': {
            'precision_macro': 0.56, 'precision_micro': 0.40, 'precision_samples': 0.18,
            'recall_macro': 0.59, 'recall_micro': 0.40, 'recall_samples': 0.74,
            'f1_macro': 0.39, 'f1_micro': 0.40, 'f1_samples': 0.29,
            'roc_auc_macro': 0.65, 'pr_auc_macro': 0.27
          }
        },
        '8': {
          'baseline': {
            'precision_macro': 0.62, 'precision_micro': 0.45, 'precision_samples': 0.23,
            'recall_macro': 0.64, 'recall_micro': 0.45, 'recall_samples': 0.78,
            'f1_macro': 0.45, 'f1_micro': 0.45, 'f1_samples': 0.33,
            'roc_auc_macro': 0.69, 'pr_auc_macro': 0.32
          }
        },
        '9': {
          'baseline': {
            'precision_macro': 0.57, 'precision_micro': 0.41, 'precision_samples': 0.20,
            'recall_macro': 0.60, 'recall_micro': 0.41, 'recall_samples': 0.75,
            'f1_macro': 0.41, 'f1_micro': 0.41, 'f1_samples': 0.30,
            'roc_auc_macro': 0.66, 'pr_auc_macro': 0.29
          }
        }
      }
    }
  };

  console.log('Mock data created');
  initializeCharts();
}

// Initialize charts
function initializeCharts() {
  const taskLabels = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6', 'Task 7', 'Task 8', 'Task 9'];

  // Colors for different metric types
  const colors = {
    macro: 'rgba(59, 130, 246, 1)',
    micro: 'rgba(16, 185, 129, 1)',
    samples: 'rgba(245, 158, 11, 1)',
    roc: 'rgba(139, 92, 246, 1)',
    pr: 'rgba(239, 68, 68, 1)'
  };

  // Initialize precision chart
  const precisionCtx = document.getElementById('precisionChart').getContext('2d');
  charts.precision = new Chart(precisionCtx, {
    type: 'line',
    data: {
      labels: taskLabels,
      datasets: [
        {
          label: 'Precision (Macro)',
          data: getMetricData('precision', 'macro'),
          borderColor: colors.macro,
          backgroundColor: colors.macro.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.macro,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        },
        {
          label: 'Precision (Micro)',
          data: getMetricData('precision', 'micro'),
          borderColor: colors.micro,
          backgroundColor: colors.micro.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.micro,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        },
        {
          label: 'Precision (Samples)',
          data: getMetricData('precision', 'samples'),
          borderColor: colors.samples,
          backgroundColor: colors.samples.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.samples,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: false,
          text: 'Precision Metrics Across Tasks',
          font: {
            size: 18,
            weight: 'bold'
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          title: {
            display: true,
            text: 'Score'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Tasks'
          }
        }
      }
    }
  });

  // Initialize recall chart
  const recallCtx = document.getElementById('recallChart').getContext('2d');
  charts.recall = new Chart(recallCtx, {
    type: 'line',
    data: {
      labels: taskLabels,
      datasets: [
        {
          label: 'Recall (Macro)',
          data: getMetricData('recall', 'macro'),
          borderColor: colors.macro,
          backgroundColor: colors.macro.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.macro,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        },
        {
          label: 'Recall (Micro)',
          data: getMetricData('recall', 'micro'),
          borderColor: colors.micro,
          backgroundColor: colors.micro.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.micro,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        },
        {
          label: 'Recall (Samples)',
          data: getMetricData('recall', 'samples'),
          borderColor: colors.samples,
          backgroundColor: colors.samples.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.samples,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: false,
          text: 'Recall Metrics Across Tasks',
          font: {
            size: 18,
            weight: 'bold'
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          title: {
            display: true,
            text: 'Score'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Tasks'
          }
        }
      }
    }
  });

  // Initialize F1 chart
  const f1Ctx = document.getElementById('f1Chart').getContext('2d');
  charts.f1 = new Chart(f1Ctx, {
    type: 'line',
    data: {
      labels: taskLabels,
      datasets: [
        {
          label: 'F1 Score (Macro)',
          data: getMetricData('f1', 'macro'),
          borderColor: colors.macro,
          backgroundColor: colors.macro.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.macro,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        },
        {
          label: 'F1 Score (Micro)',
          data: getMetricData('f1', 'micro'),
          borderColor: colors.micro,
          backgroundColor: colors.micro.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.micro,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        },
        {
          label: 'F1 Score (Samples)',
          data: getMetricData('f1', 'samples'),
          borderColor: colors.samples,
          backgroundColor: colors.samples.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.samples,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: false,
          text: 'F1 Score Metrics Across Tasks',
          font: {
            size: 18,
            weight: 'bold'
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          title: {
            display: true,
            text: 'Score'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Tasks'
          }
        }
      }
    }
  });

  // Initialize AUC chart
  const aucCtx = document.getElementById('aucChart').getContext('2d');
  charts.auc = new Chart(aucCtx, {
    type: 'line',
    data: {
      labels: taskLabels,
      datasets: [
        {
          label: 'ROC AUC',
          data: getMetricData('auc', 'roc'),
          borderColor: colors.roc,
          backgroundColor: colors.roc.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.roc,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        },
        {
          label: 'PR AUC',
          data: getMetricData('auc', 'pr'),
          borderColor: colors.pr,
          backgroundColor: colors.pr.replace('1)', '0.1)'),
          borderWidth: 3,
          fill: false,
          tension: 0.4,
          pointBackgroundColor: colors.pr,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: false,
          text: 'AUC Metrics Across Tasks',
          font: {
            size: 18,
            weight: 'bold'
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          title: {
            display: true,
            text: 'Score'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Tasks'
          }
        }
      }
    }
  });
}

// Get metric data for a specific metric type
function getMetricData(metric, type) {
  const modelData = forwardData[currentModel];
  const taskData = modelData.per_task;
  const data = [];

  for (let task = 1; task <= 9; task++) {
    const taskKey = task.toString();
    if (taskData[taskKey] && taskData[taskKey].baseline) {
      const baseline = taskData[taskKey].baseline;

      if (metric === 'auc') {
        if (type === 'roc') {
          data.push(baseline.roc_auc_macro);
        } else if (type === 'pr') {
          data.push(baseline.pr_auc_macro);
        }
      } else {
        const metricKey = `${metric}_${type}`;
        data.push(baseline[metricKey]);
      }
    } else {
      data.push(0);
    }
  }

  return data;
}


// Update chart data
function updateChartData(metric) {
  const chart = charts[metric];

  if (metric === 'precision') {
    // Update precision chart data
    chart.data.datasets[0].data = getMetricData('precision', 'macro');
    chart.data.datasets[1].data = getMetricData('precision', 'micro');
    chart.data.datasets[2].data = getMetricData('precision', 'samples');
  } else if (metric === 'recall') {
    // Update recall chart data
    chart.data.datasets[0].data = getMetricData('recall', 'macro');
    chart.data.datasets[1].data = getMetricData('recall', 'micro');
    chart.data.datasets[2].data = getMetricData('recall', 'samples');
  } else if (metric === 'f1') {
    // Update F1 chart data
    chart.data.datasets[0].data = getMetricData('f1', 'macro');
    chart.data.datasets[1].data = getMetricData('f1', 'micro');
    chart.data.datasets[2].data = getMetricData('f1', 'samples');
  } else if (metric === 'auc') {
    // Update AUC chart data
    chart.data.datasets[0].data = getMetricData('auc', 'roc');
    chart.data.datasets[1].data = getMetricData('auc', 'pr');
  }

  chart.update();
}

// Update charts when model changes
function updateCharts() {
  currentModel = document.getElementById('model-select').value;
  console.log('Model changed to:', currentModel);

  // Update all charts
  Object.keys(charts).forEach(metric => {
    updateChartData(metric);
  });

  // Update table if table view is active
  const tableView = document.getElementById('table-view');
  if (tableView.classList.contains('active')) {
    populateTable();
  }

  // Update per-class table if tab active
  const perClassTab = document.getElementById('per-class');
  if (perClassTab.classList.contains('active')) {
    populatePerClassF1Table();
  }
}

// Tab functionality
function openTab(tabName) {
  // Hide all tab content
  const tabContents = document.querySelectorAll('.tab-content');
  tabContents.forEach(tab => tab.classList.remove('active'));

  // Remove active class from all tab buttons
  const tabButtons = document.querySelectorAll('.tab-button');
  tabButtons.forEach(button => button.classList.remove('active'));

  // Show selected tab content
  document.getElementById(tabName).classList.add('active');

  // Add active class to clicked button
  event.target.classList.add('active');

  if (tabName === 'dataset') {
    ensureDatasetInitialized();
  }
  if (tabName === 'per-class') {
    populatePerClassF1Table();
  }
  if (tabName === 'train-config') {
    populateConfigViews();
  }
}

// View switching functionality
function showView(viewName) {
  // Hide all view content
  const viewContents = document.querySelectorAll('.view-content');
  viewContents.forEach(view => view.classList.remove('active'));

  // Remove active class from all view buttons
  const viewButtons = document.querySelectorAll('.view-button');
  viewButtons.forEach(button => button.classList.remove('active'));

  // Show selected view content
  document.getElementById(viewName + '-view').classList.add('active');

  // Add active class to clicked button
  event.target.classList.add('active');

  // Populate table if switching to table view
  if (viewName === 'table') {
    populateTable();
  }
  if (viewName === 'per-class') {
    populatePerClassF1Table();
  }
}

// Configuration view switching functionality
function showConfigView(viewName) {
  // Hide all config view content
  const configViewContents = document.querySelectorAll('#train-config .view-content');
  configViewContents.forEach(view => view.classList.remove('active'));

  // Remove active class from all config view buttons
  const configViewButtons = document.querySelectorAll('#train-config .view-button');
  configViewButtons.forEach(button => button.classList.remove('active'));

  // Show selected config view content
  document.getElementById(viewName + '-view').classList.add('active');

  // Add active class to clicked button
  event.target.classList.add('active');
}

// Populate configuration views
function populateConfigViews() {
  populateConfigSummaries();
  populateConfigDetails();
}

// Populate configuration summaries
function populateConfigSummaries() {
  const models = ['replay-30', 'replay-60', 'ewc-2000'];

  models.forEach(model => {
    const config = trainingConfigs[model];
    const summaryEl = document.getElementById(`${model}-summary`);

    if (!summaryEl) return;

    let summaryHtml = `
      <div class="config-highlight">
        <div class="highlight-title">Key Differences:</div>
        <div class="highlight-content">${config.key_differences.join(', ')}</div>
      </div>
      <div style="margin-top: 15px; padding: 10px; background: #f1f5f9; border-radius: 6px;">
        <div style="margin-bottom: 6px;"><strong>🔄 Strategy:</strong> ${config.use_exemplars} exemplars</div>
        <div style="margin-bottom: 6px;"><strong>⏱️ Training:</strong> ${config.num_epochs} epochs, LR=${config.learning_rate}</div>
        <div style="margin-bottom: 6px;"><strong>⚙️ Loss:</strong> ${config.loss_type.toUpperCase()} (C=${config.rldam_C})</div>
        <div style="margin-bottom: 6px;"><strong>🚀 Performance:</strong> FP16=${config.fp16}, Memory Opt=${config.gradient_checkpointing}</div>
    `;

    if (config.ewc_lambda !== null && config.ewc_lambda !== 'N/A') {
      summaryHtml += `<div style="margin-bottom: 6px;"><strong>⚖️ EWC:</strong> λ=${config.ewc_lambda} regularization</div>`;
    }

    if (config.adaptive_training) {
      summaryHtml += `<div style="margin-bottom: 6px;"><strong>🎯 Adaptive:</strong> <span style="color: #f59e0b;">Enabled</span></div>`;
    }

    summaryHtml += `</div>`;
    summaryEl.innerHTML = summaryHtml;
  });
}

// Populate detailed configuration tables
function populateConfigDetails() {
  // First populate the comparison and common config tables
  populateComparisonTable();
  populateCommonConfigTable();

  const models = ['replay-30', 'replay-60', 'ewc-2000'];

  models.forEach(model => {
    const config = trainingConfigs[model];

    // Populate model architecture settings
    populateModelSettings(model, config);

    // Populate training settings
    populateTrainingSettings(model, config);

    // Populate performance & memory settings
    populatePerformanceSettings(model, config);

    // Handle adaptive training methods for EWC
    if (model === 'ewc-2000' && config.adaptive_training) {
      populateAdaptiveMethodsTable(config);
      populateAdaptiveRulesTable(config);
    }
  });
}

// Populate strategy comparison table
function populateComparisonTable() {
  const comparisonBodyEl = document.getElementById('comparison-body');

  if (!comparisonBodyEl) return;

  comparisonBodyEl.innerHTML = '';

  const strategies = [
    {
      name: 'Replay-30',
      exemplars: '30%',
      ewc_lambda: 'None',
      epochs: 15,
      features: 'Standard replay with 30% exemplar buffer'
    },
    {
      name: 'Replay-60',
      exemplars: '60%',
      ewc_lambda: 'None',
      epochs: 15,
      features: 'Enhanced replay with 60% exemplar buffer'
    },
    {
      name: 'EWC-2000',
      exemplars: 'None',
      ewc_lambda: '2000',
      epochs: 15,
      features: 'Adaptive training with EWC regularization'
    }
  ];

  strategies.forEach(strategy => {
    const row = document.createElement('tr');

    const nameCell = document.createElement('td');
    nameCell.textContent = strategy.name;
    nameCell.className = 'param-name';

    const exemplarsCell = document.createElement('td');
    exemplarsCell.textContent = strategy.exemplars;
    exemplarsCell.className = 'param-value';

    const ewcCell = document.createElement('td');
    ewcCell.textContent = strategy.ewc_lambda;
    ewcCell.className = 'param-value';

    const epochsCell = document.createElement('td');
    epochsCell.textContent = strategy.epochs;
    epochsCell.className = 'param-value';

    const featuresCell = document.createElement('td');
    featuresCell.textContent = strategy.features;
    featuresCell.className = 'param-desc';

    row.appendChild(nameCell);
    row.appendChild(exemplarsCell);
    row.appendChild(ewcCell);
    row.appendChild(epochsCell);
    row.appendChild(featuresCell);

    comparisonBodyEl.appendChild(row);
  });
}

// Populate common configuration table
function populateCommonConfigTable() {
  const commonBodyEl = document.getElementById('common-config-body');

  if (!commonBodyEl) return;

  commonBodyEl.innerHTML = '';

  const commonSettings = [
    ['Model Architecture', 'GraphCodeBERT', 'microsoft/graphcodebert-base'],
    ['Loss Function', 'RLDAM', 'C=1.0, nth_power=0.25'],
    ['Training Data', 'Dataset', 'norm-source (9 tasks)'],
    ['Base LR', 'Learning Rate', '2e-5'],
    ['Batch Size', 'Train/Eval', '32/32'],
    ['Max Length', 'Sequence Length', '512 tokens'],
    ['Memory', 'Gradient Checkpointing', 'Enabled'],
    ['Precision', 'Mixed Precision', 'FP16 Enabled'],
    ['Workers', 'Data Loading', '4 workers'],
    ['Logging', 'Strategy', 'Every epoch']
  ];

  commonSettings.forEach(([category, setting, value]) => {
    const row = document.createElement('tr');

    const categoryCell = document.createElement('td');
    categoryCell.textContent = category;
    categoryCell.className = 'param-name';

    const settingCell = document.createElement('td');
    settingCell.textContent = setting;
    settingCell.className = 'param-value';

    const valueCell = document.createElement('td');
    valueCell.textContent = value;
    valueCell.className = 'param-desc';

    row.appendChild(categoryCell);
    row.appendChild(settingCell);
    row.appendChild(valueCell);

    commonBodyEl.appendChild(row);
  });
}

// Populate model architecture settings
function populateModelSettings(model, config) {
  const modelBodyEl = document.getElementById(`${model}-model-body`);

  if (!modelBodyEl) return;

  modelBodyEl.innerHTML = '';

  // All strategies share the same model architecture
  const modelSettings = [
    ['model_name', config.model_name, 'Base model architecture'],
    ['tokenizer_path', 'Pre-trained tokenizer', 'Tokenizer path (common)'],
    ['max_length', config.max_length, 'Maximum sequence length'],
    ['max_chunks', config.max_chunks, 'Maximum code chunks to process'],
    ['dropout_rate', config.dropout_rate, 'Dropout for regularization'],
    ['chunk_aggregation_method', config.chunk_aggregation_method, 'How chunks are combined']
  ];

  modelSettings.forEach(([param, value, desc]) => {
    const row = document.createElement('tr');

    const paramCell = document.createElement('td');
    paramCell.textContent = param;
    paramCell.className = 'param-name';

    const valueCell = document.createElement('td');
    valueCell.textContent = value;
    valueCell.className = 'param-value';

    const descCell = document.createElement('td');
    descCell.textContent = desc;
    descCell.className = 'param-desc';

    row.appendChild(paramCell);
    row.appendChild(valueCell);
    row.appendChild(descCell);

    modelBodyEl.appendChild(row);
  });
}

// Populate training settings
function populateTrainingSettings(model, config) {
  const trainingBodyEl = document.getElementById(`${model}-training-body`);

  if (!trainingBodyEl) return;

  trainingBodyEl.innerHTML = '';

  // Strategy-specific training settings
  const strategySpecificSettings = [];

  if (model.includes('replay')) {
    strategySpecificSettings.push(
      ['use_exemplars', config.use_exemplars, 'Exemplar buffer size'],
      ['exemplar_selection_strategy', config.exemplar_selection_strategy, 'How exemplars are selected'],
      ['exemplar_combine_strategy', config.exemplar_combine_strategy, 'How to combine old and new data']
    );
  } else if (model.includes('ewc')) {
    strategySpecificSettings.push(
      ['ewc_lambda', config.ewc_lambda, 'EWC regularization strength'],
      ['adaptive_training', config.adaptive_training ? 'Enabled' : 'Disabled', 'Adaptive hyperparameters by task size']
    );
  }

  // Common training settings
  const commonSettings = [
    ['loss_type', config.loss_type, 'RLDAM loss function'],
    ['rldam_C', config.rldam_C, 'RLDAM scaling parameter'],
    ['rldam_nth_power', config.rldam_nth_power, 'RLDAM power parameter'],
    ['num_epochs', config.num_epochs, 'Training epochs per task'],
    ['learning_rate', config.learning_rate, 'Base learning rate'],
    ['warmup_ratio', config.warmup_ratio, 'Learning rate warmup ratio'],
    ['weight_decay', config.weight_decay, 'L2 regularization'],
    ['early_stopping_patience', config.early_stopping_patience, 'Early stopping tolerance']
  ];

  // Add strategy-specific settings first
  strategySpecificSettings.forEach(([param, value, desc]) => {
    const row = document.createElement('tr');
    row.style.backgroundColor = '#e0f2fe'; // Light blue background for strategy-specific

    const paramCell = document.createElement('td');
    paramCell.textContent = param;
    paramCell.className = 'param-name';

    const valueCell = document.createElement('td');
    valueCell.textContent = value;
    valueCell.className = 'param-value';

    const descCell = document.createElement('td');
    descCell.textContent = desc;
    descCell.className = 'param-desc';

    row.appendChild(paramCell);
    row.appendChild(valueCell);
    row.appendChild(descCell);

    trainingBodyEl.appendChild(row);
  });

  // Add common settings
  commonSettings.forEach(([param, value, desc]) => {
    const row = document.createElement('tr');

    const paramCell = document.createElement('td');
    paramCell.textContent = param;
    paramCell.className = 'param-name';

    const valueCell = document.createElement('td');
    valueCell.textContent = value;
    valueCell.className = 'param-value';

    const descCell = document.createElement('td');
    descCell.textContent = desc;
    descCell.className = 'param-desc';

    row.appendChild(paramCell);
    row.appendChild(valueCell);
    row.appendChild(descCell);

    trainingBodyEl.appendChild(row);
  });
}

// Populate performance & memory settings
function populatePerformanceSettings(model, config) {
  const performanceBodyEl = document.getElementById(`${model}-performance-body`);

  if (!performanceBodyEl) return;

  performanceBodyEl.innerHTML = '';

  // Core performance settings (common)
  const coreSettings = [
    ['gradient_checkpointing', config.gradient_checkpointing ? 'Enabled' : 'Disabled', 'Memory optimization'],
    ['fp16', config.fp16 ? 'Enabled' : 'Disabled', 'Mixed precision training'],
    ['auto_find_batch_size', config.auto_find_batch_size ? 'Enabled' : 'Disabled', 'Automatic batch size optimization'],
    ['dataloader_num_workers', config.dataloader_num_workers, 'Data loading workers'],
    ['dataloader_pin_memory', config.dataloader_pin_memory ? 'Enabled' : 'Disabled', 'Faster GPU transfer']
  ];

  // Strategy-specific settings
  const strategySettings = [];

  if (model.includes('replay')) {
    strategySettings.push(
      ['exemplar_selection_strategy', config.exemplar_selection_strategy, 'Exemplar selection method'],
      ['exemplar_combine_strategy', config.exemplar_combine_strategy, 'Data combination strategy'],
      ['exemplar_unmask_strategy', config.exemplar_unmask_strategy, 'Class masking approach']
    );
  }

  // Infrastructure settings
  const infraSettings = [
    ['dataset_name', config.dataset_name, 'Training dataset'],
    ['num_tasks', config.num_tasks, 'Total continual learning tasks'],
    ['output_base_dir', config.output_base_dir, 'Model save location'],
    ['resume_from_task', config.resume_from_task || 'None', 'Resume capability']
  ];

  // Add section headers and content
  addSettingsSection(performanceBodyEl, 'Core Performance', coreSettings, '#10b981');
  addSettingsSection(performanceBodyEl, 'Strategy-Specific', strategySettings, '#f59e0b');
  addSettingsSection(performanceBodyEl, 'Infrastructure', infraSettings, '#3b82f6');
}

// Helper function to add settings sections with headers
function addSettingsSection(parentEl, sectionTitle, settings, headerColor) {
  if (settings.length === 0) return;

  // Add section header
  const headerRow = document.createElement('tr');
  const headerCell = document.createElement('td');
  headerCell.colSpan = 3;
  headerCell.innerHTML = `<h6 style="color: ${headerColor}; margin: 10px 0; font-size: 0.9rem; font-weight: 600;">${sectionTitle}</h6>`;
  headerCell.style.padding = '8px';
  headerCell.style.backgroundColor = '#f8fafc';
  headerRow.appendChild(headerCell);
  parentEl.appendChild(headerRow);

  // Add settings
  settings.forEach(([param, value, desc]) => {
    const row = document.createElement('tr');

    const paramCell = document.createElement('td');
    paramCell.textContent = param;
    paramCell.className = 'param-name';

    const valueCell = document.createElement('td');
    valueCell.textContent = value;
    valueCell.className = 'param-value';

    const descCell = document.createElement('td');
    descCell.textContent = desc;
    descCell.className = 'param-desc';

    row.appendChild(paramCell);
    row.appendChild(valueCell);
    row.appendChild(descCell);

    parentEl.appendChild(row);
  });
}

// Populate adaptive methods table for EWC
function populateAdaptiveMethodsTable(config) {
  const adaptiveBodyEl = document.getElementById('ewc-2000-adaptive-body');

  if (!adaptiveBodyEl) return;

  adaptiveBodyEl.innerHTML = '';

  const adaptiveMethods = [
    ['Early Stopping Patience', config.adaptive_methods.patience, 'Reduces patience for small tasks to avoid overfitting'],
    ['Learning Rate', config.adaptive_methods.learning_rate, 'Increases LR for small tasks to learn faster'],
    ['Warmup Ratio', config.adaptive_methods.warmup, 'Reduces warmup for very small tasks'],
    ['Batch Size', config.adaptive_methods.batch_size, 'Uses smaller batches for tiny tasks'],
    ['EWC Lambda', config.adaptive_methods.ewc_lambda, 'Reduces EWC constraint for small tasks'],
    ['Min Epochs', config.adaptive_methods.min_epochs, 'Forces minimum training for small tasks']
  ];

  adaptiveMethods.forEach(([method, logic, desc]) => {
    const row = document.createElement('tr');

    const methodCell = document.createElement('td');
    methodCell.textContent = method;
    methodCell.className = 'param-name';

    const logicCell = document.createElement('td');
    logicCell.innerHTML = formatLogicCode(logic);
    logicCell.className = 'param-value';

    const descCell = document.createElement('td');
    descCell.textContent = desc;
    descCell.className = 'param-desc';

    row.appendChild(methodCell);
    row.appendChild(logicCell);
    row.appendChild(descCell);

    adaptiveBodyEl.appendChild(row);
  });
}

// Format logic code for better display
function formatLogicCode(logic) {
  const lines = logic.split('\n');
  const formattedLines = lines.map(line => {
    const trimmed = line.trim();
    if (trimmed.startsWith('if') || trimmed.startsWith('elif') || trimmed.startsWith('else')) {
      return `<span class="keyword">${trimmed}</span>`;
    } else if (trimmed.includes('return')) {
      return `<span class="return-value">${trimmed}</span>`;
    } else {
      return `<span class="comment">${trimmed}</span>`;
    }
  });

  return `<div class="logic-code">${formattedLines.join('<br>')}</div>`;
}

// Populate adaptive rules table for EWC
function populateAdaptiveRulesTable(config) {
  const rulesBodyEl = document.getElementById('ewc-2000-rules-body');

  if (!rulesBodyEl) return;

  rulesBodyEl.innerHTML = '';

  config.adaptive_rules.forEach(rule => {
    const row = document.createElement('tr');

    const rangeCell = document.createElement('td');
    const adjustmentsCell = document.createElement('td');

    // Split the rule on the first colon to separate range from adjustments
    const colonIndex = rule.indexOf(':');
    if (colonIndex > 0) {
      rangeCell.textContent = rule.substring(0, colonIndex).trim();
      adjustmentsCell.innerHTML = formatRuleAdjustments(rule.substring(colonIndex + 1).trim());
    } else {
      rangeCell.textContent = 'All Tasks';
      adjustmentsCell.innerHTML = formatRuleAdjustments(rule);
    }

    rangeCell.className = 'param-name';
    adjustmentsCell.className = 'param-desc';

    row.appendChild(rangeCell);
    row.appendChild(adjustmentsCell);

    rulesBodyEl.appendChild(row);
  });
}

// Format rule adjustments with color coding
function formatRuleAdjustments(adjustments) {
  // Split by commas and format each parameter
  return adjustments.split(', ').map(adjustment => {
    const parts = adjustment.trim().split('=');
    if (parts.length === 2) {
      const param = parts[0].trim();
      const value = parts[1].trim();
      return `<span style="color: #059669; font-weight: 500;">${param}</span>=<span style="color: #dc2626; font-weight: 600;">${value}</span>`;
    }
    return adjustment;
  }).join(', ');
}

// Populate table with baseline data
function populateTable() {
  const tableBody = document.getElementById('table-body');
  const modelData = forwardData[currentModel];
  const taskData = modelData.per_task;

  // Clear existing table data
  tableBody.innerHTML = '';

  // Populate table with data for each task
  for (let task = 1; task <= 9; task++) {
    const taskKey = task.toString();
    if (taskData[taskKey] && taskData[taskKey].baseline) {
      const baseline = taskData[taskKey].baseline;

      const row = document.createElement('tr');
      // Store task index on row
      row.dataset.task = String(task);

      // Task cell
      const taskCell = document.createElement('td');
      taskCell.textContent = `Task ${task}`;
      taskCell.className = 'task-cell';
      row.appendChild(taskCell);

      // Precision metrics
      const precisionMacro = document.createElement('td');
      precisionMacro.textContent = formatValue(baseline.precision_macro);
      precisionMacro.dataset.metric = 'precision_macro';
      precisionMacro.dataset.value = String(baseline.precision_macro);
      row.appendChild(precisionMacro);

      const precisionMicro = document.createElement('td');
      precisionMicro.textContent = formatValue(baseline.precision_micro);
      precisionMicro.dataset.metric = 'precision_micro';
      precisionMicro.dataset.value = String(baseline.precision_micro);
      row.appendChild(precisionMicro);

      const precisionSamples = document.createElement('td');
      precisionSamples.textContent = formatValue(baseline.precision_samples);
      precisionSamples.dataset.metric = 'precision_samples';
      precisionSamples.dataset.value = String(baseline.precision_samples);
      row.appendChild(precisionSamples);

      // Recall metrics
      const recallMacro = document.createElement('td');
      recallMacro.textContent = formatValue(baseline.recall_macro);
      recallMacro.dataset.metric = 'recall_macro';
      recallMacro.dataset.value = String(baseline.recall_macro);
      row.appendChild(recallMacro);

      const recallMicro = document.createElement('td');
      recallMicro.textContent = formatValue(baseline.recall_micro);
      recallMicro.dataset.metric = 'recall_micro';
      recallMicro.dataset.value = String(baseline.recall_micro);
      row.appendChild(recallMicro);

      const recallSamples = document.createElement('td');
      recallSamples.textContent = formatValue(baseline.recall_samples);
      recallSamples.dataset.metric = 'recall_samples';
      recallSamples.dataset.value = String(baseline.recall_samples);
      row.appendChild(recallSamples);

      // F1 metrics
      const f1Macro = document.createElement('td');
      f1Macro.textContent = formatValue(baseline.f1_macro);
      f1Macro.dataset.metric = 'f1_macro';
      f1Macro.dataset.value = String(baseline.f1_macro);
      row.appendChild(f1Macro);

      const f1Micro = document.createElement('td');
      f1Micro.textContent = formatValue(baseline.f1_micro);
      f1Micro.dataset.metric = 'f1_micro';
      f1Micro.dataset.value = String(baseline.f1_micro);
      row.appendChild(f1Micro);

      const f1Samples = document.createElement('td');
      f1Samples.textContent = formatValue(baseline.f1_samples);
      f1Samples.dataset.metric = 'f1_samples';
      f1Samples.dataset.value = String(baseline.f1_samples);
      row.appendChild(f1Samples);

      // AUC metrics
      const rocAuc = document.createElement('td');
      rocAuc.textContent = formatValue(baseline.roc_auc_macro);
      rocAuc.dataset.metric = 'roc_auc_macro';
      rocAuc.dataset.value = String(baseline.roc_auc_macro);
      row.appendChild(rocAuc);

      const prAuc = document.createElement('td');
      prAuc.textContent = formatValue(baseline.pr_auc_macro);
      prAuc.dataset.metric = 'pr_auc_macro';
      prAuc.dataset.value = String(baseline.pr_auc_macro);
      row.appendChild(prAuc);

      tableBody.appendChild(row);
    }
  }

  // Attach hover listeners depending on mode
  if (hoverMode === 'forgetting' && forgettingData && forgettingData[currentModel] && forgettingData[currentModel].backward_transfer) {
    attachBackwardTransferHover(tableBody, forgettingData[currentModel].backward_transfer, taskData);
  } else {
    attachForwardTransferHover(tableBody, taskData);
  }
}

// Format numeric values for display
function formatValue(value) {
  if (value === null || value === undefined) {
    return 'N/A';
  }
  return (typeof value === 'number') ? value.toFixed(4) : value;
}

// Toggle calculation info section
function toggleCalculationInfo() {
  const content = document.getElementById('calculation-info-content');
  const toggle = document.querySelector('.info-toggle');

  if (content.style.display === 'none' || content.style.display === '') {
    content.style.display = 'block';
    toggle.innerHTML = 'Hide Details ▲';
  } else {
    content.style.display = 'none';
    toggle.innerHTML = 'Show Details ▼';
  }
}

function formatDelta(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  const sign = value > 0 ? '+' : value < 0 ? '-' : '+';
  return sign + Math.abs(value).toFixed(4);
}

/**
 * Delta calculation:
 * - For thresholded metrics: forward_transfer[i -> j, metric] - baseline[i, metric]
 *   (Fixed-threshold drift: compares forward transfer with same-threshold baseline)
 * - For AUC/PR-AUC: baseline[j, metric] - forward_transfer[i -> j, metric]
 *   (Gap to j-optimized: how far from the j-tuned baseline)
 * Positive: better performance, Negative: worse performance
 */

function attachForwardTransferHover(tableBody, taskData) {
  const rows = Array.from(tableBody.querySelectorAll('tr'));

  function clearForwardDeltas() {
    const currentRows = Array.from(tableBody.querySelectorAll('tr'));
    currentRows.forEach(row => {
      const cells = Array.from(row.querySelectorAll('td'));
      for (let c = 1; c < cells.length; c++) {
        const td = cells[c];
        td.classList.remove('delta-positive', 'delta-negative');
        const baseVal = td.dataset.value;
        if (baseVal !== undefined) {
          const num = Number(baseVal);
          td.textContent = formatValue(Number.isNaN(num) ? baseVal : num);
        }
      }
    });
  }

  function applyForwardDeltas(hoverTaskIndex) {
    // Ensure clean slate
    clearForwardDeltas();

    const hoverKey = String(hoverTaskIndex);
    const hoverTask = taskData[hoverKey];
    if (!hoverTask || !hoverTask.forward_transfer) return;

    // For each subsequent row j > i
    for (let j = hoverTaskIndex + 1; j <= 9; j++) {
      const row = rows[j - 1]; // rows are 1-indexed by task
      if (!row) continue;
      const forwardForJ = hoverTask.forward_transfer[String(j)];
      if (!forwardForJ) continue;

      const cells = Array.from(row.querySelectorAll('td'));
      // cells[0] is Task label; metrics start from 1 aligned with METRIC_COLUMN_KEYS
      for (let col = 0; col < METRIC_COLUMN_KEYS.length; col++) {
        const td = cells[col + 1];
        if (!td) continue;
        const metricKey = METRIC_COLUMN_KEYS[col];
        const fwdVal = forwardForJ[metricKey];

        // For AUC metrics, compare with baseline at task j (gap to j-optimized)
        // For thresholded metrics, use fixed-threshold drift (compare with baseline at task i)
        let baseVal;
        let delta;
        if (metricKey === 'roc_auc_macro' || metricKey === 'pr_auc_macro') {
          // AUC mode: gap to j-optimized
          const taskJBaseline = taskData[String(j)]?.baseline;
          baseVal = taskJBaseline ? taskJBaseline[metricKey] : NaN;
          delta = baseVal - fwdVal; // Negative means forward is better (smaller gap)
        } else {
          // Fixed-threshold mode: drift from i baseline
          const baseValStr = td.dataset.value;
          baseVal = baseValStr !== undefined ? Number(baseValStr) : NaN;
          delta = fwdVal - baseVal;
        }

        if (typeof fwdVal === 'number' && typeof baseVal === 'number' && !Number.isNaN(baseVal)) {
          td.textContent = formatDelta(delta);
          if (delta > 0) {
            td.classList.add('delta-positive');
            td.classList.remove('delta-negative');
          } else if (delta < 0) {
            td.classList.add('delta-negative');
            td.classList.remove('delta-positive');
          } else {
            td.classList.remove('delta-positive', 'delta-negative');
          }
        } else {
          td.textContent = 'N/A';
          td.classList.remove('delta-positive', 'delta-negative');
        }
      }
    }
  }

  // Bind listeners
  rows.forEach(row => {
    row.addEventListener('mouseenter', () => {
      if (hoverMode !== 'forward') return;
      const taskIndex = Number(row.dataset.task);
      if (!Number.isNaN(taskIndex)) {
        applyForwardDeltas(taskIndex);
      }
    });
  });

  if (!tableBody.dataset.forwardHoverBound) {
    tableBody.addEventListener('mouseleave', () => {
      clearForwardDeltas();
    });
    tableBody.dataset.forwardHoverBound = '1';
  }
}

function attachPerClassForwardHover(tableBody, taskData) {
  const rows = Array.from(tableBody.querySelectorAll('tr'));
  // Headers remain uncolored; we only color data cells

  function clearPerClassDeltas() {
    const currentRows = Array.from(tableBody.querySelectorAll('tr'));
    currentRows.forEach(row => {
      const cells = Array.from(row.querySelectorAll('td'));
      // skip first column (vulnerability label) and last column (avg)
      for (let c = 1; c < cells.length - 1; c++) {
        const td = cells[c];
        td.classList.remove('delta-positive', 'delta-negative');
        const baseVal = td.dataset.value;
        if (baseVal !== undefined) {
          const num = Number(baseVal);
          td.textContent = formatValue(Number.isNaN(num) ? baseVal : num);
        }
      }
    });
    // no-op for headers
  }

  function applyPerClassDeltas(hoverTaskIndex) {
    clearPerClassDeltas();
    const hoverKey = String(hoverTaskIndex);
    const hoverTask = taskData[hoverKey];
    if (!hoverTask || !hoverTask.forward_transfer) return;

    // We do not color headers; only cells

    for (let j = hoverTaskIndex + 1; j <= 9; j++) {
      // Iterate all vulnerability rows
      rows.forEach(vRow => {
        const cells = Array.from(vRow.querySelectorAll('td'));
        // vulnerability label at index 0, tasks 1..9, avg last
        const td = cells[j];
        if (!td) return;
        const vulnType = td.dataset.vuln;
        if (!vulnType) return;

        const forwardForJ = hoverTask.forward_transfer[String(j)];
        if (!forwardForJ) return;

        const metricKey = `f1_${vulnType}`;
        const baseValStr = td.dataset.value;
        const baseVal = baseValStr !== undefined ? Number(baseValStr) : NaN;
        const fwdVal = forwardForJ[metricKey];
        if (typeof fwdVal === 'number' && typeof baseVal === 'number' && !Number.isNaN(baseVal)) {
          const delta = fwdVal - baseVal;
          td.textContent = formatDelta(delta);
          if (delta > 0) {
            td.classList.add('delta-positive');
            td.classList.remove('delta-negative');
          } else if (delta < 0) {
            td.classList.add('delta-negative');
            td.classList.remove('delta-positive');
          } else {
            td.classList.remove('delta-positive', 'delta-negative');
          }
        } else {
          td.textContent = 'N/A';
          td.classList.remove('delta-positive', 'delta-negative');
        }
      });
      // headers unchanged
    }
  }

  // Bind listeners: hover over any task cell in any row triggers deltas
  rows.forEach(row => {
    const cells = Array.from(row.querySelectorAll('td'));
    for (let c = 1; c <= 9; c++) {
      const td = cells[c];
      if (!td) continue;
      td.addEventListener('mouseenter', () => {
        if (hoverMode !== 'forward') return;
        const taskIndex = Number(td.dataset.task);
        if (!Number.isNaN(taskIndex)) {
          applyPerClassDeltas(taskIndex);
        }
      });
    }
  });

  if (!tableBody.dataset.perClassHoverBound) {
    tableBody.addEventListener('mouseleave', () => {
      clearPerClassDeltas();
    });
    tableBody.dataset.perClassHoverBound = '1';
  }
}

function attachBackwardTransferHover(tableBody, backwardTransfer, taskData) {
  const rows = Array.from(tableBody.querySelectorAll('tr'));

  function clearBwdDeltas() {
    const currentRows = Array.from(tableBody.querySelectorAll('tr'));
    currentRows.forEach(row => {
      const cells = Array.from(row.querySelectorAll('td'));
      for (let c = 1; c < cells.length; c++) {
        const td = cells[c];
        td.classList.remove('delta-positive', 'delta-negative');
        const baseVal = td.dataset.value;
        if (baseVal !== undefined) {
          const num = Number(baseVal);
          td.textContent = formatValue(Number.isNaN(num) ? baseVal : num);
        }
      }
    });
  }

  function applyBwdDeltas(currentTaskIndex) {
    clearBwdDeltas();
    const jKey = String(currentTaskIndex);
    const bwdForJ = backwardTransfer[jKey];
    if (!bwdForJ || !bwdForJ.previous_tasks) return;

    // For each previous task i < j, display current(j→i) - baseline(i)
    for (let i = 1; i < currentTaskIndex; i++) {
      const row = rows[i - 1];
      if (!row) continue;
      const prevTaskEntry = bwdForJ.previous_tasks[String(i)];
      if (!prevTaskEntry || !prevTaskEntry.current || !prevTaskEntry.baseline) continue;

      const cells = Array.from(row.querySelectorAll('td'));
      for (let col = 0; col < METRIC_COLUMN_KEYS.length; col++) {
        const td = cells[col + 1];
        if (!td) continue;
        const metricKey = METRIC_COLUMN_KEYS[col];
        const curVal = prevTaskEntry.current[metricKey];
        const baseVal = prevTaskEntry.baseline[metricKey];
        if (typeof curVal === 'number' && typeof baseVal === 'number') {
          const delta = curVal - baseVal;
          td.textContent = formatDelta(delta);
          if (delta > 0) {
            td.classList.add('delta-positive');
            td.classList.remove('delta-negative');
          } else if (delta < 0) {
            td.classList.add('delta-negative');
            td.classList.remove('delta-positive');
          } else {
            td.classList.remove('delta-positive', 'delta-negative');
          }
        } else {
          td.textContent = 'N/A';
          td.classList.remove('delta-positive', 'delta-negative');
        }
      }
    }
  }

  // Bind listeners
  rows.forEach(row => {
    row.addEventListener('mouseenter', () => {
      if (hoverMode !== 'forgetting') return;
      const taskIndex = Number(row.dataset.task);
      if (!Number.isNaN(taskIndex)) {
        applyBwdDeltas(taskIndex);
      }
    });
  });

  if (!tableBody.dataset.backwardHoverBound) {
    tableBody.addEventListener('mouseleave', () => {
      clearBwdDeltas();
    });
    tableBody.dataset.backwardHoverBound = '1';
  }
}

// Per-class F1 table from forward (baseline) per_task
function populatePerClassF1Table() {
  const tableBody = document.getElementById('per-class-f1-body');
  if (!tableBody || !forwardData) return;

  const modelData = forwardData[currentModel];
  if (!modelData || !modelData.per_task) return;

  const taskData = modelData.per_task;
  const vulnerabilityLabels = [
    'ACCESS_CONTROL', 'ARITHMETIC', 'BAD_RANDOMNESS', 'CODE_QUALITY', 'CONTRACT_LIFECYCLE',
    'DATA_STORAGE', 'DELEGATE_CALL', 'DENIAL_OF_SERVICE', 'ERROR_HANDLING', 'FRONT_RUNNING',
    'GAS_ISSUES', 'HONEYPOT', 'OTHER', 'REENTRANCY',
    'STANDARDS', 'TIME_MANIPULATION', 'TX_ORIGIN', 'UNCHECKED_CALLS'
  ];

  tableBody.innerHTML = '';

  vulnerabilityLabels.forEach(vulnType => {
    const row = document.createElement('tr');

    const vulnCell = document.createElement('td');
    vulnCell.textContent = formatVulnerabilityName(vulnType);
    vulnCell.className = 'vulnerability-cell';
    row.appendChild(vulnCell);

    let sum = 0;
    let count = 0;

    for (let task = 1; task <= 9; task++) {
      const taskKey = String(task);
      const td = document.createElement('td');
      if (taskData[taskKey] && taskData[taskKey].baseline) {
        const baseline = taskData[taskKey].baseline;
        const key = `f1_${vulnType}`;
        const value = baseline[key];
        if (typeof value === 'number') {
          td.textContent = value.toFixed(4);
          td.dataset.vuln = vulnType;
          td.dataset.task = String(task);
          td.dataset.value = String(value);
          sum += value;
          count += 1;
        } else {
          td.textContent = 'N/A';
          td.className = 'no-data-cell';
        }
      } else {
        td.textContent = 'N/A';
        td.className = 'no-data-cell';
      }
      row.appendChild(td);
    }

    const avg = document.createElement('td');
    if (count > 0) {
      const mean = sum / count;
      avg.textContent = mean.toFixed(4);
      avg.className = 'average-cell';
    } else {
      avg.textContent = 'N/A';
      avg.className = 'average-cell no-data-cell';
    }
    row.appendChild(avg);

    tableBody.appendChild(row);
  });

  if (hoverMode === 'forgetting' && forgettingData && forgettingData[currentModel] && forgettingData[currentModel].backward_transfer) {
    attachPerClassBackwardHover(tableBody, forgettingData[currentModel].backward_transfer);
  } else {
    attachPerClassForwardHover(tableBody, taskData);
  }
}

function formatVulnerabilityName(v) {
  const shortNames = {
    'ACCESS_CONTROL': 'Access Control',
    'ARITHMETIC': 'Arithmetic',
    'BAD_RANDOMNESS': 'Bad Randomness',
    'CODE_QUALITY': 'Code Quality',
    'CONTRACT_LIFECYCLE': 'Lifecycle',
    'DATA_STORAGE': 'Data Storage',
    'DELEGATE_CALL': 'Delegate Call',
    'DENIAL_OF_SERVICE': 'DoS',
    'ERROR_HANDLING': 'Error Handling',
    'FRONT_RUNNING': 'Front Running',
    'GAS_ISSUES': 'Gas Issues',
    'HONEYPOT': 'Honeypot',
    'OTHER': 'Other',
    'REENTRANCY': 'Reentrancy',
    'REPLAY_ATTACK': 'Replay Attack',
    'SHORT_ADDRESS': 'Short Address',
    'STANDARDS': 'Standards',
    'TIME_MANIPULATION': 'Time Manipulation',
    'TX_ORIGIN': 'Tx Origin',
    'UNCHECKED_CALLS': 'Unchecked Calls'
  };
  return shortNames[v] || v.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatVulnCountsForTooltip(vulnCounts) {
  if (!vulnCounts) return '';
  const nonZero = Object.entries(vulnCounts).filter(([_, count]) => count > 0);
  if (nonZero.length === 0) return '';

  // Sort by count descending for better readability
  nonZero.sort((a, b) => b[1] - a[1]);

  // Show ALL vulnerabilities without truncation
  const allVulns = nonZero.map(([name, count]) => `${formatVulnerabilityName(name)}: ${count}`);
  return `Vulnerabilities (${nonZero.length} types):\n${allVulns.join('\n')}`;
}

function getF1ScoreClass(f) {
  if (f >= 0.7) return 'high-score';
  if (f >= 0.5) return 'medium-score';
  if (f >= 0.3) return 'medium-score';
  return 'low-score';
}

function attachPerClassBackwardHover(tableBody, backwardTransfer) {
  const rows = Array.from(tableBody.querySelectorAll('tr'));

  function clearPerClassDeltas() {
    const currentRows = Array.from(tableBody.querySelectorAll('tr'));
    currentRows.forEach(row => {
      const cells = Array.from(row.querySelectorAll('td'));
      // skip first col (label) and last col (avg)
      for (let c = 1; c < cells.length - 1; c++) {
        const td = cells[c];
        td.classList.remove('delta-positive', 'delta-negative');
        const baseVal = td.dataset.value;
        if (baseVal !== undefined) {
          const num = Number(baseVal);
          td.textContent = formatValue(Number.isNaN(num) ? baseVal : num);
        }
      }
    });
  }

  function applyPerClassBwdDeltas(currentTaskIndex) {
    clearPerClassDeltas();
    const jKey = String(currentTaskIndex);
    const bwdForJ = backwardTransfer[jKey];
    if (!bwdForJ || !bwdForJ.previous_tasks) return;

    // For each previous task i < j, update that column's cells across all vuln rows
    rows.forEach(vRow => {
      const cells = Array.from(vRow.querySelectorAll('td'));
      for (let i = 1; i < currentTaskIndex; i++) {
        const td = cells[i];
        if (!td) continue;
        const vulnType = td.dataset.vuln;
        if (!vulnType) continue;
        const prevTaskEntry = bwdForJ.previous_tasks[String(i)];
        if (!prevTaskEntry) continue;
        const curVal = prevTaskEntry.current[`f1_${vulnType}`];
        const baseVal = prevTaskEntry.baseline[`f1_${vulnType}`];
        if (typeof curVal === 'number' && typeof baseVal === 'number') {
          const delta = curVal - baseVal;
          td.textContent = formatDelta(delta);
          if (delta > 0) {
            td.classList.add('delta-positive');
            td.classList.remove('delta-negative');
          } else if (delta < 0) {
            td.classList.add('delta-negative');
            td.classList.remove('delta-positive');
          } else {
            td.classList.remove('delta-positive', 'delta-negative');
          }
        } else {
          td.textContent = 'N/A';
          td.classList.remove('delta-positive', 'delta-negative');
        }
      }
    });
  }

  // Bind listeners on each task column cell
  rows.forEach(row => {
    const cells = Array.from(row.querySelectorAll('td'));
    for (let c = 1; c <= 9; c++) {
      const td = cells[c];
      if (!td) continue;
      td.addEventListener('mouseenter', () => {
        if (hoverMode !== 'forgetting') return;
        const taskIndex = Number(td.dataset.task);
        if (!Number.isNaN(taskIndex)) {
          applyPerClassBwdDeltas(taskIndex);
        }
      });
    }
  });

  if (!tableBody.dataset.perClassBwdHoverBound) {
    tableBody.addEventListener('mouseleave', () => {
      clearPerClassDeltas();
    });
    tableBody.dataset.perClassBwdHoverBound = '1';
  }
}

function setHoverMode(mode, toggleId) {
  hoverMode = mode;
  // Toggle button active state in all hover toggles
  const toggleContainers = document.querySelectorAll('.hover-toggle');
  toggleContainers.forEach(container => {
    const buttons = container.querySelectorAll('.hover-button');
    buttons.forEach(btn => btn.classList.remove('active'));
    const toActivate = Array.from(buttons).find(b => b.textContent.toLowerCase().includes(mode === 'forward' ? 'forward' : 'forgetting'));
    if (toActivate) toActivate.classList.add('active');
  });

  // Update footnote text
  const footnote = document.getElementById('table-footnote-text');
  if (footnote) {
    if (hoverMode === 'forward') {
      footnote.textContent = 'AUC columns use "gap to j-optimized" mode (baseline[j] - forward_transfer[i→j]) to show closeness to j-tuned performance';
    } else {
      footnote.textContent = 'Forgetting deltas show current(j→i) - baseline(i) using thresholds chosen on task i for all metrics (including AUCs)';
    }
  }

  // Rebind hover for currently visible views
  const tableView = document.getElementById('table-view');
  if (tableView && tableView.classList.contains('active')) {
    populateTable();
  }
  const perClassTab = document.getElementById('per-class');
  if (perClassTab && perClassTab.classList.contains('active')) {
    populatePerClassF1Table();
  }
}
