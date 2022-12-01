import unittest
from collections import OrderedDict
import torch.nn as nn

from o2grad.recursive import (
    get_o2modules,
    remove_o2modules,
    set_next_layer,
    replace_with_o2modules,
)
from o2grad.modules.o2loss import O2Loss
from o2grad.modules.o2layer.o2layer import O2Layer
from o2grad.modules.o2parametric.o2linear import O2Linear
from o2grad.modules.o2parametric.o2conv1d import O2Conv1d
from o2grad.modules.o2container.o2sequential import O2Sequential
from o2grad.modules.o2container.o2residual import Residual, O2Residual


class RecursiveTest(unittest.TestCase):
    def _setup(self):
        model = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(5, 5)),
                    ("relu", nn.ReLU()),
                    ("fc2", nn.Linear(5, 5)),
                    ("seq", nn.Sequential(nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 5))),
                    ("res", Residual(nn.Linear(5, 5), nn.ReLU())),
                ]
            )
        )
        return model

    def test_set_next_layer(self):
        residual = O2Residual(O2Linear(5, 5), O2Linear(5, 5), O2Linear(5, 5))
        model = O2Sequential(
            OrderedDict(
                [
                    ("fc1", O2Linear(5, 5)),
                    ("fc2", residual),
                    ("fc3", O2Linear(5, 5)),
                ]
            )
        )
        criterion = O2Loss(nn.MSELoss())
        set_next_layer(model, criterion)
        self.assertEqual(id(model.next_layer), id(criterion))
        self.assertEqual(id(model.module.fc1.next_layer), id(model.module.fc2))
        self.assertEqual(id(model.module.fc2.next_layer), id(model.module.fc3))
        self.assertEqual(
            id(residual.module.sequence[0].next_layer), id(residual.module.sequence[1])
        )
        self.assertEqual(
            id(residual.module.sequence[1].next_layer), id(residual.module.sequence[2])
        )
        self.assertEqual(
            id(residual.module.sequence[2].next_layer), id(model.module.fc3)
        )

    def test_get_o2modules(self):
        model = O2Sequential(O2Linear(5, 5), O2Linear(5, 5))
        o2modules = get_o2modules(model)
        self.assertTrue(len(o2modules) == 2)
        are_o2modules = [isinstance(m, O2Layer) for m in o2modules.values()]
        self.assertTrue(all(are_o2modules))
        # Nested with residual
        model = O2Sequential(
            OrderedDict(
                [
                    ("fc1", O2Linear(5, 5)),
                    (
                        "residual",
                        O2Residual(
                            O2Linear(5, 5),
                            O2Linear(5, 5),
                        ),
                    ),
                    ("fc2", O2Linear(5, 5)),
                ]
            )
        )
        o2modules = get_o2modules(model)
        self.assertTrue(len(o2modules) == 4)
        are_o2modules = [isinstance(m, O2Layer) for m in o2modules.values()]
        self.assertTrue(all(are_o2modules))

    def test_replace_with_o2modules(self):
        # Wrapping sequential
        model = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
        o2model = replace_with_o2modules(model)
        self.assertTrue(isinstance(o2model, O2Sequential))
        self.assertTrue(isinstance(o2model[0], O2Linear))
        self.assertTrue(isinstance(o2model[1], O2Linear))
        # Wrapping residual
        model = Residual(nn.Linear(5, 5), nn.Linear(5, 5))
        o2model = replace_with_o2modules(model)
        self.assertTrue(isinstance(o2model, O2Residual))
        self.assertTrue(isinstance(o2model.module.sequence, O2Sequential))
        self.assertTrue(isinstance(o2model.module.sequence[0], O2Linear))
        self.assertTrue(isinstance(o2model.module.sequence[1], O2Linear))
        # Wrapping mixed
        model = nn.Sequential(
            nn.Linear(5, 5),
            O2Linear(5, 5),
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3),
        )
        o2model = replace_with_o2modules(model)
        self.assertTrue(isinstance(o2model, O2Sequential))
        self.assertTrue(isinstance(o2model[0], O2Linear))
        self.assertTrue(isinstance(o2model[1], O2Linear))
        self.assertTrue(isinstance(o2model[2], O2Conv1d))
        # Wrapping nested
        model = Residual(
            nn.Linear(5, 5), Residual(nn.Linear(5, 5), nn.Linear(5, 5)), nn.Linear(5, 3)
        )
        o2model = replace_with_o2modules(model)
        self.assertTrue(isinstance(o2model, O2Residual))
        self.assertTrue(isinstance(o2model.module.sequence, O2Sequential))
        self.assertTrue(isinstance(o2model.module.sequence[0], O2Linear))
        self.assertTrue(isinstance(o2model.module.sequence[1], O2Residual))
        self.assertTrue(
            isinstance(o2model.module.sequence[1].module.sequence[0], O2Linear)
        )
        self.assertTrue(
            isinstance(o2model.module.sequence[1].module.sequence[1], O2Linear)
        )
        self.assertTrue(isinstance(o2model.module.sequence[2], O2Linear))
        # Wrapping nested
        model = Residual(Residual(nn.Linear(5, 5), nn.Linear(5, 5)))
        o2model = replace_with_o2modules(model)
        self.assertTrue(isinstance(o2model, O2Residual))
        self.assertTrue(isinstance(o2model.module.sequence, O2Residual))
        self.assertTrue(
            isinstance(o2model.module.sequence.module.sequence[0], O2Linear)
        )
        self.assertTrue(
            isinstance(o2model.module.sequence.module.sequence[1], O2Linear)
        )

    def test_remove_o2modules(self):
        model = Residual(
            nn.Linear(5, 5), Residual(nn.Linear(5, 5), nn.Linear(5, 5)), nn.Linear(5, 3)
        )
        o2model = replace_with_o2modules(model)
        model = remove_o2modules(o2model)
        print(model)
        self.assertTrue(isinstance(model, Residual))
        self.assertTrue(isinstance(model.sequence[0], nn.Linear))
        self.assertTrue(isinstance(model.sequence[1], Residual))
        self.assertTrue(isinstance(model.sequence[1].sequence[0], nn.Linear))
        self.assertTrue(isinstance(model.sequence[1].sequence[1], nn.Linear))
        self.assertTrue(isinstance(model.sequence[2], nn.Linear))
