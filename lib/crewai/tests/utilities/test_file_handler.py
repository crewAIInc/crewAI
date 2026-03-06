import json
import os
import pickle
import unittest
import uuid

from crewai.utilities.file_handler import PickleHandler


class TestPickleHandler(unittest.TestCase):
    def setUp(self):
        # Use a unique file name for each test to avoid race conditions in parallel test execution
        unique_id = str(uuid.uuid4())
        self.file_name = f"test_data_{unique_id}"
        self.json_path = os.path.join(os.getcwd(), self.file_name + ".json")
        self.pkl_path = os.path.join(os.getcwd(), self.file_name + ".pkl")
        self.handler = PickleHandler(self.file_name)

    def tearDown(self):
        for path in (self.json_path, self.pkl_path):
            if os.path.exists(path):
                os.remove(path)

    def test_initialize_file(self):
        assert os.path.exists(self.json_path) is False

        self.handler.initialize_file()

        assert os.path.exists(self.json_path) is True
        assert os.path.getsize(self.json_path) >= 0

    def test_save_and_load(self):
        data = {"key": "value"}
        self.handler.save(data)
        loaded_data = self.handler.load()
        assert loaded_data == data

    def test_load_empty_file(self):
        loaded_data = self.handler.load()
        assert loaded_data == {}

    def test_load_corrupted_file(self):
        """Test that corrupted (non-JSON) files return empty dict gracefully."""
        with open(self.json_path, "w") as file:
            file.write("corrupted data that is not valid json")

        loaded_data = self.handler.load()
        assert loaded_data == {}

    def test_uses_json_format(self):
        """Test that data is saved in JSON format, not pickle."""
        data = {"agent1": {"param1": 1, "param2": "test"}}
        self.handler.save(data)

        # Verify the file is valid JSON
        with open(self.json_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_file_extension_is_json(self):
        """Test that the handler uses .json extension."""
        handler = PickleHandler("test_file.pkl")
        assert handler.file_path.endswith(".json")
        assert not handler.file_path.endswith(".pkl")

    def test_no_pickle_in_saved_file(self):
        """Test that saved files do not contain pickle data (security)."""
        data = {"key": "value", "nested": {"a": 1}}
        self.handler.save(data)

        with open(self.json_path, "rb") as f:
            raw = f.read()

        # Pickle files start with specific opcodes (0x80 for protocol 2+)
        assert not raw.startswith(b"\x80"), "File appears to contain pickle data"
        # Should be valid UTF-8 text (JSON)
        raw.decode("utf-8")

    def test_migrate_legacy_pkl_file(self):
        """Test that legacy .pkl files are automatically migrated to JSON."""
        data = {"agent1": {"param1": 1}}

        # Create a legacy pkl file
        with open(self.pkl_path, "wb") as f:
            pickle.dump(data, f)

        assert os.path.exists(self.pkl_path)
        assert not os.path.exists(self.json_path)

        # Loading should migrate the pkl to json
        loaded_data = self.handler.load()
        assert loaded_data == data

        # pkl file should be removed after migration
        assert not os.path.exists(self.pkl_path)
        # json file should now exist
        assert os.path.exists(self.json_path)

    def test_pkl_extension_input_uses_json(self):
        """Test that passing a .pkl filename still results in .json storage."""
        handler = PickleHandler("my_data.pkl")
        assert handler.file_path.endswith("my_data.json")

    def test_insecure_pickle_not_loaded_directly(self):
        """Test that arbitrary pickle files cannot be loaded directly as JSON.

        This verifies the security fix: a malicious pickle file placed at the
        JSON path would not be deserialized via pickle.load().
        """
        # Create a file with pickle content at the json path
        malicious_data = {"safe": True}
        with open(self.json_path, "wb") as f:
            pickle.dump(malicious_data, f)

        # The handler should fail gracefully (corrupt JSON) rather than
        # executing pickle.load on this file
        loaded = self.handler.load()
        assert loaded == {}  # Returns empty dict for corrupted JSON
