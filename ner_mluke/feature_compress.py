class FeatureExtracted:
    def __init__(
            self,
            list_input_ids_tensor,
            list_attention_mask_tensor,
            list_entity_ids_tensor,
            list_entity_position_ids_tensor,
            list_entity_attention_mask_tensor
    ):
        self.list_input_ids_tensor = list_input_ids_tensor
        self.list_attention_mask_tensor = list_attention_mask_tensor
        self.list_entity_ids_tensor = list_entity_ids_tensor
        self.list_entity_position_ids_tensor = list_entity_position_ids_tensor
        self.list_entity_attention_mask_tensor = list_entity_attention_mask_tensor

        # TODO remove return
        return self.list_input_ids_tensor, self.list_attention_mask_tensor, self.list_entity_ids_tensor, self.list_entity_position_ids_tensor, self.list_entity_attention_mask_tensor

