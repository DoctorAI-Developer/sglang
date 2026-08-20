from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SpecTrainingInfo:
    """Tracks spec training info for requests in a batch.

    Keys:
    - data_ids: rid -> data_id mapping
    - packed_loss_masks: data_id -> packed_loss_mask string
    - mooncake_store_keys: data_id -> list of keys
    """

    data_ids: Dict[str, str] = field(default_factory=dict)
    packed_loss_masks: Dict[str, str] = field(default_factory=dict)
    mooncake_store_keys: Dict[str, List[str]] = field(default_factory=dict)

    def add_request(
        self,
        rid: str,
        data_id: Optional[str],
        packed_loss_mask: Optional[str],
    ):
        """Add spec training info for a request if it's a spec training request."""
        if data_id is not None:
            self.data_ids[rid] = data_id
            self.packed_loss_masks[data_id] = packed_loss_mask
            if data_id not in self.mooncake_store_keys:
                self.mooncake_store_keys[data_id] = []

    def has_request(self, rid: str) -> bool:
        return rid in self.data_ids

    def set_mooncake_store_keys(self, data_id: str, keys: List[str]):
        if data_id in self.mooncake_store_keys:
            self.mooncake_store_keys[data_id] = keys

    def remove_request(self, rid: str):
        data_id = self.data_ids.pop(rid, None)
        if data_id is not None:
            remaining_rids_with_data_id = [
                r for r, d in self.data_ids.items() if d == data_id
            ]
            if not remaining_rids_with_data_id:
                self.packed_loss_masks.pop(data_id, None)
                self.mooncake_store_keys.pop(data_id, None)

    def is_empty(self) -> bool:
        return len(self.data_ids) == 0
