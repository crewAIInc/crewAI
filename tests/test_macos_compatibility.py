import pytest
import platform


class TestMacOSCompatibility:
    """Test macOS compatibility, especially onnxruntime dependency resolution."""

    def test_chromadb_import_success(self):
        """Test that ChromaDB can be imported successfully."""
        try:
            import chromadb
            assert chromadb is not None
            assert hasattr(chromadb, '__version__')
        except ImportError as e:
            pytest.fail(f"ChromaDB import failed: {e}")

    def test_onnxruntime_import_success(self):
        """Test that onnxruntime can be imported successfully."""
        try:
            import onnxruntime
            assert onnxruntime is not None
            assert hasattr(onnxruntime, '__version__')
        except ImportError as e:
            pytest.fail(f"onnxruntime import failed: {e}")

    def test_onnxruntime_version_compatibility(self):
        """Test that onnxruntime version is within expected range."""
        try:
            import onnxruntime
            version = onnxruntime.__version__
            
            major, minor, patch = map(int, version.split('.'))
            version_tuple = (major, minor, patch)
            
            min_version = (1, 19, 0)
            max_version = (1, 22, 0)
            
            assert version_tuple >= min_version, f"onnxruntime version {version} is below minimum {'.'.join(map(str, min_version))}"
            assert version_tuple <= max_version, f"onnxruntime version {version} is above maximum {'.'.join(map(str, max_version))}"
            
        except ImportError:
            pytest.skip("onnxruntime not available for version check")

    def test_chromadb_persistent_client_creation(self):
        """Test that ChromaDB PersistentClient can be created successfully."""
        try:
            from crewai.utilities.chromadb import create_persistent_client
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                client = create_persistent_client(path=temp_dir)
                assert client is not None
                
        except ImportError as e:
            pytest.fail(f"ChromaDB utilities import failed: {e}")
        except Exception as e:
            pytest.fail(f"ChromaDB client creation failed: {e}")

    def test_rag_storage_initialization(self):
        """Test that RAGStorage can be initialized successfully."""
        try:
            from crewai.memory.storage.rag_storage import RAGStorage
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                storage = RAGStorage(
                    type="test_memory",
                    allow_reset=True,
                    embedder_config=None,
                    crew=None,
                    path=temp_dir
                )
                assert storage is not None
                assert hasattr(storage, 'app')
                assert hasattr(storage, 'collection')
                
        except ImportError as e:
            pytest.fail(f"RAGStorage import failed: {e}")
        except Exception as e:
            pytest.fail(f"RAGStorage initialization failed: {e}")

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific test")
    def test_macos_onnxruntime_availability(self):
        """Test that onnxruntime is available on macOS with proper version."""
        try:
            import onnxruntime
            version = onnxruntime.__version__
            
            major, minor, patch = map(int, version.split('.'))
            
            if (major, minor) == (1, 19):
                assert patch >= 0, f"onnxruntime 1.19.x version should be >= 1.19.0, got {version}"
            elif (major, minor) == (1, 20):
                pass
            elif (major, minor) == (1, 21):
                pass
            elif (major, minor) == (1, 22):
                assert patch <= 0, f"onnxruntime 1.22.x version should be <= 1.22.0, got {version}"
            else:
                pytest.fail(f"onnxruntime version {version} is outside expected range 1.19.0-1.22.0")
                
        except ImportError:
            pytest.fail("onnxruntime should be available on macOS with the new version range")

    def test_chromadb_collection_operations(self):
        """Test basic ChromaDB collection operations work with current onnxruntime."""
        try:
            from crewai.utilities.chromadb import create_persistent_client, sanitize_collection_name
            import tempfile
            import uuid
            
            with tempfile.TemporaryDirectory() as temp_dir:
                client = create_persistent_client(path=temp_dir)
                
                collection_name = sanitize_collection_name("test_collection")
                collection = client.get_or_create_collection(name=collection_name)
                
                test_doc = "This is a test document for ChromaDB compatibility."
                test_id = str(uuid.uuid4())
                
                collection.add(
                    documents=[test_doc],
                    ids=[test_id],
                    metadatas=[{"test": True}]
                )
                
                results = collection.query(
                    query_texts=["test document"],
                    n_results=1
                )
                
                assert len(results["ids"][0]) > 0
                assert results["documents"][0][0] == test_doc
                
        except ImportError as e:
            pytest.fail(f"ChromaDB operations import failed: {e}")
        except Exception as e:
            pytest.fail(f"ChromaDB operations failed: {e}")
