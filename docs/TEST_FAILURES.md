# Test Failures and Fixes Needed

## Test Run Results
- Total: 71 tests
- Passed: 57 (80%)
- Failed: 14 (20%)

## Failures to Fix

### 1. StepLog Buffer Overflow (test_lmcleaner_core.py)
**Issue**: StepLog.get() returns None for step_id=2 after overflow
**Root Cause**: step_map index becomes invalid after buffer rotation
**Fix**: Need to verify StepLog implementation handles index updates correctly

### 2. Flatten Empty List (test_lmcleaner_core.py)
**Issue**: torch.cat() fails with empty list
**Fix**: Add guard in _flatten() to return empty tensor for empty input
```python
if not tensors:
    return torch.tensor([])
```

### 3. HVP Computation Tests (test_lmcleaner_core.py)
**Issue**: Model forward() returns object with loss=None, causing TypeError in autograd.grad
**Root Cause**: Mock model doesn't compute actual loss
**Fix**: Update mock model to compute real loss:
```python
def forward(self, input_ids, labels=None, **kwargs):
    x = self.linear1(input_ids.float())
    logits = self.linear2(x)
    loss = None
    if labels is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return type('Output', (), {'logits': logits, 'loss': loss})()
```

### 4. Apply Correction Size Mismatch (test_lmcleaner_core.py)
**Issue**: RuntimeError instead of ValueError
**Fix**: Update test to expect RuntimeError or add size check before view_as()

### 5. BatchReconstructor API (test_training_logger.py)
**Issue**: __init__() doesn't accept batch_size parameter
**Fix**: Check actual BatchReconstructor signature and update tests

### 6. TrainingLogger Save/Load (test_training_logger.py)
**Issue**:
- step_log.pkl not created
- sample_indices_per_step has fewer entries than expected
**Fix**: Check actual save_to_disk() implementation and file naming

### 7. Error Handling Test (test_lmcleaner_trainers.py)
**Issue**: No exception raised for missing training log directory
**Fix**: Update test to match actual behavior (logs warning instead of raising)

## Priority

1. **High**: HVP computation tests (blocks core functionality testing)
2. **High**: TrainingLogger save/load (critical for integration)
3. **Medium**: StepLog buffer overflow (edge case)
4. **Medium**: BatchReconstructor API (need to match actual implementation)
5. **Low**: Flatten empty list (edge case, easy fix)
6. **Low**: Error handling (behavior clarification)

## Next Steps

1. Read actual implementation code to understand behavior
2. Fix mock objects to match real interfaces
3. Update test expectations to match actual behavior
4. Re-run tests and verify all pass
5. Update documentation with actual coverage numbers
