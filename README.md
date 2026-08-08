## Results

ResNet18 (transfer learning, ImageNet-pretrained, only final layer fine-tuned)
achieved 95.0% accuracy on the held-out test set after 5 epochs, with training
loss decreasing steadily each epoch (0.334 → 0.146). Precision on the "ok"
class was 0.90 while recall was 0.97, meaning the model rarely misses an
actually-ok casting but occasionally misclassifies a defective one as ok —
the riskier direction of error for a real quality-control use case.

## What I'd improve

- Address the precision/recall imbalance — a defective casting slipping
  through as "ok" is more costly than a false alarm; could tune the decision
  threshold or use a weighted loss to penalize that error type more heavily.
- Unfreeze and fine-tune more of ResNet18's layers (not just the final one)
  with a lower learning rate, which often improves accuracy further once the
  new layer has already converged.
- Try a second architecture (e.g. EfficientNet or a deeper ResNet) and
  compare, same as comparing Random Forest vs Logistic Regression in Project 1.
- Add Grad-CAM visualization to show _which part_ of an image the model
  focused on — valuable for building trust in a real QC deployment.
- CI pipeline (GitHub Actions) to run tests automatically on every push.
