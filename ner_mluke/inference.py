# from mymodel import Inference
#
# model_name_or_path = None
# inference = Inference(model_name_or_path=model_name_or_path)
#
# text = ""
# inference.infer(text)



## TODO using an infer script

# model.eval()
# with torch.set_grad_enabled(False):
#     val_loss_list = []
#     for step, val_batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
#         val_input_ids, val_attention_mask, val_entity_ids, val_entity_position_ids, val_entity_attention_mask, val_label_id = val_batch  # label_mask,
#
#         val_input_ids = torch.tensor(val_input_ids, dtype=torch.long)
#         val_attention_mask = torch.tensor(val_attention_mask, dtype=torch.long)
#         val_entity_ids = torch.tensor(val_entity_ids, dtype=torch.long)
#         val_entity_position_ids = torch.tensor(val_entity_position_ids, dtype=torch.long)
#         val_entity_attention_mask = torch.tensor(val_entity_attention_mask, dtype=torch.long)
#         val_label_id = torch.tensor(val_label_id, dtype=torch.long)
#
#         outputs = model(input_ids=val_input_ids,
#                         attention_mask=val_attention_mask,
#                         entity_ids=val_entity_ids,
#                         entity_position_ids=val_entity_position_ids,
#                         entity_attention_mask=val_entity_attention_mask,
#                         labels=val_label_id
#                         )
#
#         val_loss = outputs.loss
#         val_loss_list.append(val_loss.item())
#
#         if step % 10 == 9:
#             print('validation loss: ', val_loss_list[-1], '   step: ', step, '   epoch: ', epoch + 1)