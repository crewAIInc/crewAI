import os
import shutil
import stat
import tempfile
import unittest
import uuid
from unittest.mock import patch

import pytest
from crewai.utilities.file_handler import PickleHandler


class TestPickleHandler(unittest.TestCase):
    def setUp(self):
        self._home_patcher = patch.dict(os.environ, {})
        self._home_patcher.start()

        self._tmp_home = tempfile.mkdtemp(prefix="crewai_test_home_")
        self._home_patch = patch("os.path.expanduser", return_value=self._tmp_home)
        self._home_patch.start()

        unique_id = str(uuid.uuid4())
        self.file_name = f"test_data_{unique_id}.pkl"
        self.file_path = os.path.join(os.getcwd(), self.file_name)
        self.handler = PickleHandler(self.file_name)

    def tearDown(self):
        self._home_patch.stop()
        self._home_patcher.stop()

        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        sig_path = self.file_path + ".sig"
        if os.path.exists(sig_path):
            os.remove(sig_path)

        import shutil

        shutil.rmtree(self._tmp_home, ignore_errors=True)

    def test_initialize_file(self):
        assert os.path.exists(self.file_path) is False

        self.handler.initialize_file()

        assert os.path.exists(self.file_path) is True
        assert os.path.getsize(self.file_path) >= 0

    def test_save_and_load(self):
        data = {"key": "value"}
        self.handler.save(data)
        loaded_data = self.handler.load()
        assert loaded_data == data

    def test_save_creates_signature_file(self):
        data = {"key": "value"}
        self.handler.save(data)
        sig_path = self.file_path + ".sig"
        assert os.path.exists(sig_path)
        assert os.path.getsize(sig_path) == 32  # SHA-256 digest size

    def test_load_empty_file(self):
        loaded_data = self.handler.load()
        assert loaded_data == {}

    def test_load_corrupted_file_without_signature(self):
        with open(self.file_path, "wb") as file:
            file.write(b"corrupted data")
            file.flush()
            os.fsync(file.fileno())

        with pytest.raises(ValueError, match="no signature file found"):
            self.handler.load()

    def test_load_tampered_file_raises_error(self):
        data = {"key": "value"}
        self.handler.save(data)

        with open(self.file_path, "wb") as f:
            f.write(b"tampered pickle data")

        with pytest.raises(ValueError, match="signature mismatch"):
            self.handler.load()

    def test_load_unsigned_file_rejected(self):
        import pickle

        with open(self.file_path, "wb") as f:
            pickle.dump({"legacy": True}, f)

        with pytest.raises(ValueError, match="no signature file found"):
            self.handler.load()

    def test_overwrite_preserves_signature(self):
        data1 = {"first": True}
        self.handler.save(data1)
        loaded1 = self.handler.load()
        assert loaded1 == data1

        data2 = {"second": True}
        self.handler.save(data2)
        loaded2 = self.handler.load()
        assert loaded2 == data2

    def test_initialize_file_creates_valid_signature(self):
        self.handler.initialize_file()
        sig_path = self.file_path + ".sig"
        assert os.path.exists(sig_path)

        loaded_data = self.handler.load()
        assert loaded_data == {}

    def test_load_rejects_disappearing_signature(self):
        """If the signature file is removed after the existence check, load should raise ValueError."""
        data = {"key": "value"}
        self.handler.save(data)

        original_exists = os.path.exists
        sig_path = self.file_path + ".sig"

        def remove_sig_after_check(path):
            if path == sig_path and original_exists(path):
                os.remove(sig_path)
                return True  # exists() must report True so open() hits FileNotFoundError
            return original_exists(path)

        with patch("os.path.exists", side_effect=remove_sig_after_check):
            with pytest.raises(ValueError, match="signature file"):
                self.handler.load()

    def test_validate_key_storage_hardens_existing_insecure_directory(self):
        """A pre-existing 0755 key directory is tightened to 0700, not rejected."""
        if os.name != "posix":
            self.skipTest("POSIX-only directory hardening")
        key_dir = tempfile.mkdtemp(prefix="crewai_key_")
        key_path = os.path.join(key_dir, "key.bin")
        os.chmod(key_dir, 0o755)
        try:
            self.assertTrue(PickleHandler._validate_key_storage(key_dir, key_path))
            self.assertEqual(stat.S_IMODE(os.stat(key_dir).st_mode), 0o700)
        finally:
            os.chmod(key_dir, 0o700)
            os.rmdir(key_dir)

    def test_validate_key_storage_still_rejects_unhardenable_directory(self):
        """If tightening an insecure directory is impossible, validation fails closed."""
        if os.name != "posix":
            self.skipTest("POSIX-only directory hardening")
        key_dir = tempfile.mkdtemp(prefix="crewai_key_")
        key_path = os.path.join(key_dir, "key.bin")
        os.chmod(key_dir, 0o755)
        try:
            with patch("os.chmod", side_effect=OSError("read-only filesystem")):
                with self.assertRaises(PermissionError):
                    PickleHandler._validate_key_storage(key_dir, key_path)
        finally:
            os.chmod(key_dir, 0o700)
            os.rmdir(key_dir)

    def test_key_creation_passes_binary_flag_to_os_open(self):
        """Key creation must pass O_BINARY to os.open where the platform provides it."""
        fresh_home = tempfile.mkdtemp(prefix="crewai_test_home_")
        try:
            with (
                patch("os.path.expanduser", return_value=fresh_home),
                patch.object(os, "O_BINARY", 0x8000, create=True),
                patch("os.open", wraps=os.open) as mock_open,
            ):
                PickleHandler("binary_flag_check.pkl")
            seen_flags = [call.args[1] for call in mock_open.call_args_list]
            self.assertTrue(
                any(flags & 0x8000 for flags in seen_flags),
                "os.open must be called with O_BINARY when creating the key file",
            )
        finally:
            shutil.rmtree(fresh_home, ignore_errors=True)

    def test_save_serializes_payload_exactly_once(self):
        """save() must serialize into a single buffer that the HMAC then covers."""
        import pickle as pickle_mod

        real_dumps = pickle_mod.dumps
        calls = []

        def counting_dumps(obj, *args, **kwargs):
            calls.append(obj)
            return real_dumps(obj, *args, **kwargs)

        with patch("pickle.dumps", side_effect=counting_dumps):
            self.handler.save({"key": "value"})
        self.assertEqual(len(calls), 1)

    def test_save_replaces_symlink_at_destination(self):
        """A symlink planted at the pickle path is replaced, not followed."""
        if os.name != "posix":
            self.skipTest("POSIX-only symlink behavior")
        os.symlink(os.path.join(os.getcwd(), "dangling_target.pkl"), self.file_path)
        try:
            self.handler.save({"key": "value"})
            self.assertFalse(os.path.islink(self.file_path))
            self.assertEqual(self.handler.load(), {"key": "value"})
        finally:
            if os.path.islink(self.file_path):
                os.unlink(self.file_path)

    def test_load_refuses_symlinked_pickle_file(self):
        """A symlink at the pickle path must not be followed on load (O_NOFOLLOW)."""
        if os.name != "posix":
            self.skipTest("POSIX-only O_NOFOLLOW behavior")
        data = {"key": "value"}
        self.handler.save(data)

        real_pkl = self.file_path + ".real"
        real_sig = self.file_path + ".sig.real"
        os.rename(self.file_path, real_pkl)
        os.rename(self.file_path + ".sig", real_sig)
        os.symlink(real_pkl, self.file_path)
        os.symlink(real_sig, self.file_path + ".sig")
        try:
            with self.assertRaises(ValueError):
                self.handler.load()
        finally:
            for link in (self.file_path, self.file_path + ".sig"):
                if os.path.islink(link):
                    os.unlink(link)
            for path in (real_pkl, real_sig):
                if os.path.exists(path):
                    os.remove(path)

    def test_atomic_write_cleans_up_temp_file_on_failure(self):
        """A failed payload write must not leave the temp file behind."""
        dest_dir = os.getcwd()
        before = set(os.listdir(dest_dir))
        with patch("os.write", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                PickleHandler._atomic_write(os.path.join(dest_dir, "x.pkl"), b"data")
        self.assertEqual(before, set(os.listdir(dest_dir)))

    def test_load_refuses_non_regular_file(self):
        """A FIFO at the pickle path must be rejected instead of read."""
        if os.name != "posix":
            self.skipTest("POSIX-only FIFO behavior")
        fifo_path = self.file_path + ".fifo"
        os.mkfifo(fifo_path)
        try:
            with self.assertRaises(ValueError):
                PickleHandler._read_regular_file(fifo_path)
        finally:
            os.unlink(fifo_path)

    def test_validate_key_storage_skips_posix_checks_on_windows(self):
        """On Windows os.getuid() is unavailable; validation must skip POSIX checks."""
        key_dir = tempfile.mkdtemp(prefix="crewai_key_")
        key_path = os.path.join(key_dir, "key.bin")
        original_mode = stat.S_IMODE(os.stat(key_dir).st_mode)
        try:
            os.chmod(key_dir, 0o777)
            with patch("os.name", "nt"):
                self.assertTrue(PickleHandler._validate_key_storage(key_dir, key_path))
        finally:
            os.chmod(key_dir, original_mode)
            os.rmdir(key_dir)

    def test_validate_key_storage_rejects_symlinked_directory(self):
        """A symlinked key directory must be rejected before any ownership checks."""
        key_dir = "/nonexistent/key/dir"
        key_path = os.path.join(key_dir, "key.bin")
        fake_stat = os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        with patch("crewai.utilities.file_handler.os.lstat", return_value=fake_stat):
            with self.assertRaises(PermissionError):
                PickleHandler._validate_key_storage(key_dir, key_path)

    def test_validate_key_storage_rejects_symlinked_key_file(self):
        """A symlinked key file must be rejected before any ownership checks."""
        key_dir = "/nonexistent/key/dir"
        key_path = "/nonexistent/key/dir/key.bin"
        uid = os.getuid() if hasattr(os, "getuid") else 0
        dir_stat = os.stat_result((0o40700, 0, 0, 0, uid, 0, 0, 0, 0, 0))
        file_stat = os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 0, uid, 0, 0, 0, 0, 0))

        def fake_lstat(path):
            if path == key_dir:
                return dir_stat
            return file_stat

        with patch("crewai.utilities.file_handler.os.lstat", side_effect=fake_lstat), patch(
            "crewai.utilities.file_handler.os.path.exists", return_value=True
        ):
            with self.assertRaisesRegex(PermissionError, "key file must not be a symlink"):
                PickleHandler._validate_key_storage(key_dir, key_path)
