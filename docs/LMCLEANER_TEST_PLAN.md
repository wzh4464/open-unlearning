# LMCleaner Comprehensive Test Plan

## Overview
This document outlines the comprehensive testing strategy for all LMCleaner components.

## Components to Test

### 1. lmcleaner_core.py

#### 1.1 Data Structures
- **AuditRecord**
  - Initialization with all fields
  - Dict-like behavior
  - Field validation

- **StepRecord**
  - Initialization with required and optional fields
  - String representation
  - Weak reference handling

- **StepLog**
  - Add records to buffer
  - Get records by step_id
  - Buffer overflow handling (circular buffer)
  - has_range() method
  - clear() method
  - Index consistency after overflow

#### 1.2 HVP Configuration
- **HVPConfig**
  - Initialization with different modes
  - Default values
  - Device and dtype handling

#### 1.3 HVP Computation Functions
- **hvp_exact()**
  - Basic HVP computation
  - With custom parameters
  - Gradient computation correctness
  - Error handling for missing data

- **hvp_ggn()**
  - GGN approximation correctness
  - With/without labels
  - Different model outputs
  - Error handling

- **hvp_diagonal()**
  - With precomputed diag_H
  - Without precomputed diag_H
  - Correctness of diagonal approximation

- **hvp_apply()**
  - Different HVP modes
  - With batch_data
  - With batch_reconstructor
  - Error handling for missing data
  - Device placement

#### 1.4 Correction Computation
- **compute_correction()**
  - Basic correction computation
  - With different K values
  - Edge cases: K=0, K > available steps
  - Missing step records
  - Damping application
  - Audit record generation
  - HVP call counting

- **apply_correction()**
  - Correct parameter updates
  - Vector size mismatch handling
  - Multiple parameter groups

#### 1.5 Utility Functions
- **_flatten()**
  - Single tensor
  - Multiple tensors
  - Empty list

- **_unflatten_like()**
  - Correct shape restoration
  - Size mismatch handling

- **compute_param_update_vector()**
  - Correct update computation
  - Different parameter sizes

- **clone_parameters()**
  - CPU placement
  - Gradient requirement filtering

### 2. training_logger.py

#### 2.1 TrainingLogger Class
- **Initialization**
  - Different modes (batch/sample)
  - Storage options
  - Directory creation

- **register_step()**
  - Batch mode registration
  - Sample mode registration
  - With/without batch_data
  - With/without model
  - RNG state saving
  - Sample indices saving

- **_prune_old_entries()**
  - Memory management
  - Correct pruning logic
  - Edge cases

- **save_to_disk()**
  - File creation
  - Data persistence
  - Incremental saves

- **load_from_disk()**
  - Loading saved data
  - Handling missing files
  - Data integrity

#### 2.2 BatchReconstructor Class
- **Initialization**
  - Dataset and collator setup

- **get_batch_for_step()**
  - Batch reconstruction from indices
  - RNG state restoration
  - Error handling

### 3. lmcleaner_sample.py

#### 3.1 LMCleanerSampleLevel Class
- **Initialization**
  - Training log loading
  - HVP config setup
  - Audit directory creation

- **_apply_unlearning()**
  - Sample-level correction computation
  - Multiple forget samples
  - Audit record generation

- **train()**
  - Integration with base trainer
  - Immediate vs deferred application

- **_save_audit_logs()**
  - JSON export
  - File creation

### 4. lmcleaner_batch.py

#### 4.1 LMCleanerBatchLevel Class
- **Initialization**
  - Training log loading
  - HVP config setup
  - Batch reconstructor setup

- **_apply_unlearning()**
  - Batch-level correction computation
  - Multiple forget batches
  - Audit record generation

- **train()**
  - Integration with base trainer
  - Immediate vs deferred application

- **_save_audit_logs()**
  - JSON export
  - File creation

## Test Categories

### Unit Tests
- Individual function testing
- Edge case handling
- Error conditions
- Input validation

### Integration Tests
- Component interaction
- End-to-end workflows
- Data flow validation

### Performance Tests
- Memory usage
- Computation time
- Scalability

### Regression Tests
- Known bug fixes
- Critical functionality preservation

## Test Data Requirements

### Mock Models
- Small transformer model (for fast testing)
- Different parameter counts
- Different dtypes

### Mock Datasets
- Small synthetic datasets
- Known input/output pairs
- Edge case data

### Mock Training Logs
- Pre-generated step records
- Different storage modes
- Various K values

## Success Criteria

- 100% code coverage for core functions
- All edge cases handled
- No memory leaks
- Deterministic behavior
- Clear error messages
- Fast test execution (<5 min total)

## Implementation Priority

1. **High Priority** (Core functionality)
   - HVP computation functions
   - compute_correction()
   - apply_correction()
   - StepLog operations

2. **Medium Priority** (Integration)
   - TrainingLogger
   - LMCleanerSampleLevel
   - LMCleanerBatchLevel

3. **Low Priority** (Utilities)
   - Helper functions
   - Audit logging
   - File I/O
