import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import server

class DataTests(unittest.TestCase):
    def test_reject_invalid_coordinates(self):
        for x in (-.1, float('nan'), 1.1):
            with self.assertRaises(ValueError):
                server.validate_boxes([dict(class_id=0,x=x,y=0,w=.1,h=.1)])

    def test_export_requires_review_and_converts_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); photos=root/'photos';photos.mkdir()
            for i in range(3): (photos/f'{i}.png').write_bytes(bytes([i]))
            with patch.object(server,'PHOTOS',photos), patch.object(server,'DATA',root/'data'):
                with self.assertRaises(ValueError): server.export_dataset()
                for photo in server.photos()[:2]:
                    dest=server.annotation_path(photo);dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_text(json.dumps({'boxes':[dict(class_id=10,x=.2,y=.3,w=.4,h=.2)]}),'utf-8')
                result=server.export_dataset();folder=Path(result['path']).parent
                self.assertEqual((result['train'],result['val']),(1,1))
                labels=list((folder/'labels').rglob('*.txt'))
                self.assertEqual(len(labels),2)
                self.assertEqual(labels[0].read_text(),'10 0.40000000 0.40000000 0.40000000 0.20000000')

if __name__=='__main__': unittest.main()
