import base64
from pathlib import Path
import tempfile
import unittest

from save_import import import_folder, import_save, prepare_folder_import, prepare_import, split_save


def save(prefix: bytes, progress: str, suffix: bytes = b"\0") -> bytes:
    return base64.b64encode(prefix + progress.encode() + suffix)


class SaveImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.incoming = self.root / "incoming"
        self.current = self.root / "current"
        self.incoming.mkdir()
        self.current.mkdir()
        self.source = self.incoming / "progress.dat"
        self.target = self.current / "progress.dat"
        self.other = self.current / "other.dat"
        self.source.write_bytes(save(b"old-prefix{not-json}", '{"story":10,"keyUp":"W"}'))
        self.target.write_bytes(save(b"new-prefix", '{"story":0,"keyUp":"A"}', b"\0\0"))
        self.other.write_bytes(save(b"p", '{"x":1}'))
        (self.incoming / "other.dat").write_bytes(save(b"old", '{"x":99}'))

    def tearDown(self):
        self.temp.cleanup()

    def test_preserves_local_prefix_and_imports_all_json(self):
        result = prepare_import(self.source, self.target)
        self.assertEqual(base64.b64decode(result), b'new-prefix{"story":10,"keyUp":"W"}\0\0')

    def test_braces_in_binary_prefix_do_not_confuse_reader(self):
        self.assertEqual(split_save(self.source)[1], b'{"story":10,"keyUp":"W"}')

    def test_rejects_bad_and_ambiguous_input_without_changes(self):
        before = self.target.read_bytes()
        self.source.write_text("not base64")
        with self.assertRaises(ValueError):
            prepare_import(self.source, self.target)
        self.source.write_bytes(save(b"p", '{"a":1}', b" arbitrary binary suffix"))
        with self.assertRaises(ValueError):
            prepare_import(self.source, self.target)
        self.assertEqual(before, self.target.read_bytes())

    def test_backs_up_all_saves_before_replacing_only_target(self):
        backup = import_save(self.source, self.target)
        self.assertEqual((backup / "progress.dat").read_bytes(), save(b"new-prefix", '{"story":0,"keyUp":"A"}', b"\0\0"))
        self.assertEqual((backup / "other.dat").read_bytes(), self.other.read_bytes())
        self.assertEqual(base64.b64decode(self.target.read_bytes()), b'new-prefix{"story":10,"keyUp":"W"}\0\0')

    def test_missing_target_does_not_create_file_or_backup(self):
        self.target.unlink()
        with self.assertRaises(ValueError):
            import_save(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertFalse((self.current / "helper-backups").exists())

    def test_folder_import_replaces_every_matching_dat(self):
        backup, written = import_folder(self.incoming, self.current)
        self.assertCountEqual(written, ["progress.dat", "other.dat"])
        self.assertEqual(base64.b64decode(self.target.read_bytes()), b'new-prefix{"story":10,"keyUp":"W"}\0\0')
        self.assertEqual(base64.b64decode(self.other.read_bytes()), b'p{"x":99}\0')
        self.assertEqual((backup / "progress.dat").read_bytes(), save(b"new-prefix", '{"story":0,"keyUp":"A"}', b"\0\0"))
        self.assertEqual((backup / "other.dat").read_bytes(), save(b"p", '{"x":1}'))

    def test_folder_import_requires_local_targets_for_every_incoming_file(self):
        (self.incoming / "extra.dat").write_bytes(save(b"x", '{"y":1}'))
        with self.assertRaises(ValueError) as ctx:
            prepare_folder_import(self.incoming, self.current)
        self.assertIn("extra.dat", str(ctx.exception))
        self.assertEqual(self.target.read_bytes(), save(b"new-prefix", '{"story":0,"keyUp":"A"}', b"\0\0"))


if __name__ == "__main__":
    unittest.main()
