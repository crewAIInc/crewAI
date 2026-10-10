import hashlib
import hmac
import os
import pickle
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

    def test_save_writes_single_self_contained_record(self):
        """save() must store payload+signature in one atomic record, no .sig sidecar."""
        data = {"key": "value"}
        self.handler.save(data)

        sig_path = self.file_path + ".sig"
        assert not os.path.exists(sig_path)

        with open(self.file_path, "rb") as f:
            record = f.read()
        sig_len = int.from_bytes(record[:4], "big")
        assert sig_len == 32  # SHA-256 digest size
        assert (
            record[4 : 4 + 32]
            == hmac.new(self.handler._key, pickle.dumps(data), hashlib.sha256).digest()
        )
        assert self.handler.load() == data

    def test_load_reads_legacy_payload_and_sig_pair(self):
        """Files written by older versions (payload + .sig sidecar) stay readable."""
        import hmac as hmac_mod
        import hashlib as hashlib_mod
        import pickle as pickle_mod

        payload = pickle_mod.dumps({"legacy": True})
        signature = hmac_mod.new(
            self.handler._key, payload, hashlib_mod.sha256
        ).digest()
        with open(self.file_path, "wb") as f:
            f.write(payload)
        with open(self.file_path + ".sig", "wb") as f:
            f.write(signature)

        assert self.handler.load() == {"legacy": True}

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

        with open(self.file_path, "r+b") as f:
            record = bytearray(f.read())
            # Flip a byte inside the payload region of the single record.
            record[40] ^= 0xFF
            f.seek(0)
            f.write(record)

        with pytest.raises(ValueError, match="signature mismatch"):
            self.handler.load()

    def test_load_tampered_signature_prefix_raises_error(self):
        """A forged length prefix still fails HMAC verification."""
        self.handler.save({"key": "value"})

        with open(self.file_path, "r+b") as f:
            record = bytearray(f.read())
            record[0:4] = b"\x00\x00\x00\x20"
            record[4] ^= 0xFF
            f.seek(0)
            f.write(record)

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

    def test_initialize_file_creates_valid_record(self):
        self.handler.initialize_file()
        sig_path = self.file_path + ".sig"
        assert not os.path.exists(sig_path)

        loaded_data = self.handler.load()
        assert loaded_data == {}

    def test_load_rejects_disappearing_legacy_signature(self):
        """If the legacy .sig file is removed after the existence check, load raises."""
        import hmac as hmac_mod
        import hashlib as hashlib_mod
        import pickle as pickle_mod

        payload = pickle_mod.dumps({"key": "value"})
        signature = hmac_mod.new(
            self.handler._key, payload, hashlib_mod.sha256
        ).digest()
        with open(self.file_path, "wb") as f:
            f.write(payload)
        with open(self.file_path + ".sig", "wb") as f:
            f.write(signature)

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

    def test_key_creation_publishes_via_temp_file_and_link(self):
        """Key creation must stage the key in a temp file and publish with os.link,
        so a concurrent reader never observes a partially written key."""
        import tempfile as tempfile_mod

        fresh_home = tempfile.mkdtemp(prefix="crewai_test_home_")
        try:
            staged = []
            real_mkstemp = tempfile_mod.mkstemp

            def recording_mkstemp(*args, **kwargs):
                result = real_mkstemp(*args, **kwargs)
                staged.append(result[1])
                return result

            with (
                patch("os.path.expanduser", return_value=fresh_home),
                patch("os.link", wraps=os.link) as mock_link,
                patch("tempfile.mkstemp", side_effect=recording_mkstemp),
            ):
                PickleHandler("link_publish_check.pkl")
            self.assertEqual(len(staged), 1)
            self.assertTrue(
                staged[0].startswith(os.path.join(fresh_home, ".crewai") + os.sep),
                "temp key file must live in the key directory",
            )
            mock_link.assert_called_once_with(
                staged[0], os.path.join(fresh_home, ".crewai", ".hmac_key")
            )
            # The staged key must be complete (32 bytes, 0600) before publish.
            key_path = os.path.join(fresh_home, ".crewai", ".hmac_key")
            self.assertEqual(os.path.getsize(key_path), 32)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(os.stat(key_path).st_mode), 0o600)
            # No temp files may be left behind after publication.
            leftovers = [
                name
                for name in os.listdir(os.path.join(fresh_home, ".crewai"))
                if name.startswith(".hmac_key_tmp_")
            ]
            self.assertEqual(leftovers, [])
        finally:
            shutil.rmtree(fresh_home, ignore_errors=True)

    def test_key_creation_lost_link_race_loads_installed_key(self):
        """If another process publishes first (os.link raises FileExistsError),
        the installed complete key must be loaded, not the in-memory copy."""
        fresh_home = tempfile.mkdtemp(prefix="crewai_test_home_")
        try:
            key_dir = os.path.join(fresh_home, ".crewai")
            os.makedirs(key_dir, mode=0o700)
            installed_key = os.urandom(32)
            key_path = os.path.join(key_dir, ".hmac_key")
            with open(key_path, "wb") as f:
                f.write(installed_key)
            os.chmod(key_path, 0o600)

            with (
                patch("os.path.expanduser", return_value=fresh_home),
                patch("os.link", side_effect=FileExistsError),
            ):
                handler = PickleHandler("link_race_check.pkl")
            self.assertEqual(handler._key, installed_key)
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
        os.rename(self.file_path, real_pkl)
        os.symlink(real_pkl, self.file_path)
        try:
            with self.assertRaises(ValueError):
                self.handler.load()
        finally:
            if os.path.islink(self.file_path):
                os.unlink(self.file_path)
            if os.path.exists(real_pkl):
                os.remove(real_pkl)

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
