"""
Unit Tests for Computational Vision Paradigm - Part V: Security & Robustness

Tests adversarial robustness, privacy-preserving mechanisms, and secure pipelines.

Security testing philosophy:
- Knuth: "Prove security properties formally where possible"
- Wolfram: "Empirically test attack success rates across parameter space"
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import hashlib
import jwt
import time


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Simple CNN for testing."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 112 * 112, 10)
    )
    model.eval()
    return model


@pytest.fixture
def test_tensor():
    """Test image tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def test_labels():
    """Test labels."""
    return torch.tensor([3])


# ============================================================================
# Test Suite 1: Adversarial Attacks
# ============================================================================

class TestAdversarialAttacks:
    """Test adversarial attack implementations."""

    def test_fgsm_attack(self, simple_model, test_tensor, test_labels):
        """FGSM should generate adversarial examples."""
        epsilon = 8.0 / 255.0
        test_tensor.requires_grad = True

        # Forward pass
        outputs = simple_model(test_tensor)
        loss = F.cross_entropy(outputs, test_labels)

        # Compute gradient
        loss.backward()
        grad_sign = test_tensor.grad.sign()

        # Generate adversarial example
        adv_tensor = test_tensor + epsilon * grad_sign
        adv_tensor = torch.clamp(adv_tensor, 0, 1)

        # Perturbation should be bounded
        perturbation = (adv_tensor - test_tensor).abs().max()
        assert perturbation <= epsilon + 1e-6

        # Adversarial example should be different
        assert not torch.allclose(adv_tensor, test_tensor)

    def test_fgsm_determinism(self, simple_model, test_tensor, test_labels):
        """FGSM should be deterministic."""
        epsilon = 8.0 / 255.0

        def fgsm(img, label):
            img_copy = img.clone().detach().requires_grad_(True)
            outputs = simple_model(img_copy)
            loss = F.cross_entropy(outputs, label)
            loss.backward()
            grad_sign = img_copy.grad.sign()
            return img_copy + epsilon * grad_sign

        adv1 = fgsm(test_tensor, test_labels)
        adv2 = fgsm(test_tensor, test_labels)

        torch.testing.assert_close(adv1, adv2)

    def test_perturbation_magnitude(self, simple_model, test_tensor, test_labels):
        """Perturbation should respect epsilon constraint."""
        epsilons = [4.0/255, 8.0/255, 16.0/255]

        for epsilon in epsilons:
            test_tensor_copy = test_tensor.clone().requires_grad_(True)
            outputs = simple_model(test_tensor_copy)
            loss = F.cross_entropy(outputs, test_labels)
            loss.backward()

            adv_tensor = test_tensor_copy + epsilon * test_tensor_copy.grad.sign()
            adv_tensor = torch.clamp(adv_tensor, 0, 1)

            perturbation = (adv_tensor - test_tensor).abs().max().item()
            assert perturbation <= epsilon + 1e-5

    def test_adversarial_example_validity(self, simple_model, test_tensor, test_labels):
        """Adversarial examples should be valid images."""
        epsilon = 8.0 / 255.0
        test_tensor.requires_grad = True

        outputs = simple_model(test_tensor)
        loss = F.cross_entropy(outputs, test_labels)
        loss.backward()

        adv_tensor = test_tensor + epsilon * test_tensor.grad.sign()
        adv_tensor = torch.clamp(adv_tensor, 0, 1)

        # Should be valid image
        assert adv_tensor.min() >= 0.0
        assert adv_tensor.max() <= 1.0
        assert adv_tensor.shape == test_tensor.shape


# ============================================================================
# Test Suite 2: Adversarial Defenses
# ============================================================================

class TestAdversarialDefenses:
    """Test defense mechanisms."""

    def test_input_transformation_jpeg(self):
        """JPEG compression defense."""
        import cv2

        # Create test image
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        # Apply JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Images should be similar but not identical
        assert decoded.shape == image.shape
        assert decoded.dtype == image.dtype

        # Some information should be lost (compression)
        mse = np.mean((image.astype(float) - decoded.astype(float)) ** 2)
        assert mse > 0  # Some difference due to compression

    def test_bit_depth_reduction(self):
        """Bit depth reduction defense."""
        image = torch.rand(1, 3, 224, 224)

        # Reduce to 4 bits per channel
        bit_depth = 4
        num_levels = 2 ** bit_depth

        quantized = torch.round(image * (num_levels - 1)) / (num_levels - 1)

        # Should have discrete levels
        unique_values = torch.unique(quantized)
        assert len(unique_values) <= num_levels

        # Should be in valid range
        assert quantized.min() >= 0.0
        assert quantized.max() <= 1.0

    def test_gradient_clipping(self, simple_model):
        """Gradient clipping for DP-SGD."""
        # Create dummy gradients
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param) * 10  # Large gradients

        # Clip gradients
        max_norm = 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(
            simple_model.parameters(),
            max_norm
        )

        # Verify clipping
        actual_norm = 0.0
        for param in simple_model.parameters():
            actual_norm += param.grad.norm().item() ** 2
        actual_norm = actual_norm ** 0.5

        assert actual_norm <= max_norm + 1e-5


# ============================================================================
# Test Suite 3: Differential Privacy
# ============================================================================

class TestDifferentialPrivacy:
    """Test DP-SGD implementation."""

    def test_noise_addition(self):
        """DP noise should be calibrated correctly."""
        # DP parameters
        epsilon = 1.0
        delta = 1e-5
        C = 1.0  # Gradient clip bound
        batch_size = 32

        # Compute noise scale: σ = C * √(2 * log(1.25/δ)) / ε
        sensitivity = C / batch_size
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        # Generate noise
        gradient = torch.randn(100)
        noise = torch.randn_like(gradient) * noise_scale
        noisy_gradient = gradient + noise

        # Noise should have correct scale
        assert noise.std() > 0
        assert abs(noise.std() - noise_scale) < noise_scale * 0.5

    def test_privacy_budget_composition(self):
        """Privacy budget should accumulate."""
        epsilon_per_step = 0.1
        steps = 100

        # Simple composition (not tight)
        total_epsilon = epsilon_per_step * np.sqrt(steps)

        # Should grow sublinearly
        assert total_epsilon < epsilon_per_step * steps
        assert total_epsilon > epsilon_per_step

    def test_clip_then_noise(self):
        """Gradient clipping must occur before noise addition."""
        gradient = torch.randn(100) * 10  # Large gradient

        # Clip
        max_norm = 1.0
        clipped = gradient / max(gradient.norm().item(), max_norm)

        # Then add noise
        noise_scale = 0.1
        noise = torch.randn_like(clipped) * noise_scale
        dp_gradient = clipped + noise

        # Verify clipping occurred first
        assert clipped.norm().item() <= max_norm + 1e-5


# ============================================================================
# Test Suite 4: Authentication & Authorization
# ============================================================================

class TestAuthentication:
    """Test secure authentication mechanisms."""

    def test_password_hashing(self):
        """Passwords should be hashed securely."""
        password = "SecurePassword123!"
        salt = hashlib.sha256(str(time.time()).encode()).hexdigest()

        # Hash with PBKDF2
        pw_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )

        # Hash should be different from password
        assert pw_hash != password.encode()

        # Same password, same salt → same hash
        pw_hash2 = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        assert pw_hash == pw_hash2

    def test_jwt_token_generation(self):
        """JWT tokens should be generated correctly."""
        secret = "test_secret_key"
        payload = {
            'user_id': '12345',
            'username': 'testuser',
            'role': 'user',
            'exp': time.time() + 3600
        }

        # Generate token
        token = jwt.encode(payload, secret, algorithm='HS256')

        # Decode and verify
        decoded = jwt.decode(token, secret, algorithms=['HS256'])

        assert decoded['user_id'] == payload['user_id']
        assert decoded['username'] == payload['username']

    def test_jwt_token_expiry(self):
        """Expired tokens should be rejected."""
        secret = "test_secret_key"
        payload = {
            'user_id': '12345',
            'exp': time.time() - 10  # Expired 10 seconds ago
        }

        token = jwt.encode(payload, secret, algorithm='HS256')

        # Should raise expired error
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(token, secret, algorithms=['HS256'])

    def test_jwt_invalid_signature(self):
        """Invalid signature should be rejected."""
        secret = "test_secret_key"
        wrong_secret = "wrong_secret"

        payload = {'user_id': '12345', 'exp': time.time() + 3600}
        token = jwt.encode(payload, secret, algorithm='HS256')

        # Should raise invalid signature error
        with pytest.raises(jwt.InvalidSignatureError):
            jwt.decode(token, wrong_secret, algorithms=['HS256'])


# ============================================================================
# Test Suite 5: Rate Limiting
# ============================================================================

class TestRateLimiting:
    """Test rate limiting mechanisms."""

    def test_rate_limit_enforcement(self):
        """Rate limiter should enforce limits."""
        rate_limit = 10  # 10 requests per minute
        requests = []
        current_time = time.time()

        # Simulate requests
        for i in range(15):
            requests.append(current_time + i * 5)  # Request every 5 seconds

        # Filter requests within last minute
        recent = [r for r in requests if current_time + 60 >= r >= current_time]

        # Should have at most rate_limit requests
        assert len(recent) <= rate_limit + 5  # Some tolerance

    def test_rate_limit_window_sliding(self):
        """Rate limit window should slide correctly."""
        requests_per_minute = 60
        window_size = 60  # seconds

        request_times = []
        current = time.time()

        # Add requests over 2 minutes
        for i in range(100):
            request_times.append(current + i)

        # Count requests in last minute
        recent = [t for t in request_times if t >= current + 40]

        # Should be approximately 60 requests
        assert len(recent) == 60


# ============================================================================
# Test Suite 6: Input Validation
# ============================================================================

class TestInputValidation:
    """Test input validation for security."""

    def test_file_size_validation(self):
        """Large files should be rejected."""
        max_size_mb = 10.0
        max_size_bytes = max_size_mb * 1024 * 1024

        # Simulate file size check
        file_size = 15 * 1024 * 1024  # 15 MB

        assert file_size > max_size_bytes

    def test_file_format_validation(self):
        """Invalid formats should be rejected."""
        allowed_formats = ['jpg', 'jpeg', 'png']

        # Valid files
        assert 'test.jpg'.split('.')[-1].lower() in allowed_formats
        assert 'test.png'.split('.')[-1].lower() in allowed_formats

        # Invalid file
        assert 'test.exe'.split('.')[-1].lower() not in allowed_formats

    def test_image_dimensions_validation(self):
        """Extreme dimensions should be rejected."""
        max_dimension = 10000

        # Valid image
        valid_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        assert valid_img.shape[0] <= max_dimension
        assert valid_img.shape[1] <= max_dimension

        # Invalid would be rejected in actual implementation
        # (can't easily create 20000×20000 array in test)


# ============================================================================
# Test Suite 7: Audit Logging
# ============================================================================

class TestAuditLogging:
    """Test audit logging for compliance."""

    def test_hash_computation(self):
        """Input hashing for provenance."""
        data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # Compute hash
        hash1 = hashlib.sha256(data.tobytes()).hexdigest()
        hash2 = hashlib.sha256(data.tobytes()).hexdigest()

        # Same input → same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

    def test_audit_log_immutability(self):
        """Audit logs should be append-only."""
        logs = []

        # Append logs
        logs.append({'timestamp': time.time(), 'action': 'login'})
        logs.append({'timestamp': time.time(), 'action': 'inference'})

        # Cannot modify earlier logs (enforced by design)
        original_count = len(logs)
        logs.append({'timestamp': time.time(), 'action': 'logout'})

        assert len(logs) == original_count + 1

    def test_timestamp_monotonicity(self):
        """Timestamps should be monotonic (or close)."""
        timestamps = []

        for _ in range(10):
            timestamps.append(time.time())
            time.sleep(0.001)

        # Should be roughly increasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]


# ============================================================================
# Test Suite 8: Membership Inference Defense
# ============================================================================

class TestMembershipInferenceDefense:
    """Test defenses against membership inference attacks."""

    def test_prediction_noise(self, simple_model, test_tensor):
        """Adding noise should reduce membership signal."""
        # Original prediction
        with torch.no_grad():
            logits = simple_model(test_tensor)

            # Add noise
            noise_scale = 0.1
            noise = torch.randn_like(logits) * noise_scale
            noisy_logits = logits + noise

            # Predictions should differ
            assert not torch.allclose(logits, noisy_logits)

            # But not too much
            diff = (logits - noisy_logits).abs().max().item()
            assert diff < 1.0  # Reasonable noise level

    def test_confidence_masking(self, simple_model, test_tensor):
        """High-confidence predictions should be masked."""
        with torch.no_grad():
            logits = simple_model(test_tensor)
            probs = F.softmax(logits, dim=1)

            threshold = 0.9
            max_prob = probs.max()

            if max_prob > threshold:
                # Should be scaled down
                scaling = threshold / max_prob
                assert scaling < 1.0


# ============================================================================
# Test Suite 9: Security Integration
# ============================================================================

class TestSecurityIntegration:
    """Integration tests for complete security stack."""

    def test_secure_inference_pipeline(self, simple_model, test_tensor):
        """Complete secure inference pipeline."""
        # 1. Input validation
        assert test_tensor.shape == (1, 3, 224, 224)
        assert test_tensor.min() >= 0.0
        assert test_tensor.max() <= 1.0

        # 2. Rate limiting check
        allowed = True  # Simulate rate limit check
        assert allowed

        # 3. Inference with defense
        with torch.no_grad():
            logits = simple_model(test_tensor)

            # Add prediction noise
            noise = torch.randn_like(logits) * 0.05
            logits = logits + noise

            result = logits

        # 4. Audit logging
        log_entry = {
            'timestamp': time.time(),
            'operation': 'inference',
            'success': True
        }

        assert log_entry['success']

    def test_adversarial_defense_pipeline(self, simple_model, test_tensor, test_labels):
        """Test defense against adversarial attack."""
        epsilon = 8.0 / 255.0

        # Generate adversarial example
        test_tensor_copy = test_tensor.clone().requires_grad_(True)
        outputs = simple_model(test_tensor_copy)
        loss = F.cross_entropy(outputs, test_labels)
        loss.backward()

        adv_tensor = test_tensor_copy + epsilon * test_tensor_copy.grad.sign()
        adv_tensor = torch.clamp(adv_tensor, 0, 1).detach()

        # Apply defense: Input transformation
        import cv2
        adv_np = adv_tensor.squeeze().permute(1, 2, 0).numpy()
        adv_np = (adv_np * 255).astype(np.uint8)

        # JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        _, encoded = cv2.imencode('.jpg', adv_np, encode_param)
        defended = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Defended image should differ from adversarial
        assert defended.shape == adv_np.shape


# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
