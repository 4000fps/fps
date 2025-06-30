import torch
import torch.nn.functional as F

from .CLIP2Video.modules.modeling import CLIP2Video as CLIP2VideoBaseModel

NEG_INF: float = torch.finfo(torch.float32).min


class CLIP2VideoModel(CLIP2VideoBaseModel):
    def get_text_embeddings(
        self,
        sequence_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        shaped: bool = False,
    ) -> torch.Tensor:
        """Returns L2-normalized text embeddings."""
        if not shaped:
            attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])

        # Handle TAB centre loss returns (seq_out, hidden_out)
        if self.center_type == "TAB":
            sequence_output, _ = sequence_output

        if not isinstance(sequence_output, torch.Tensor):
            raise TypeError("sequence_output must be a torch.Tensor after unpacking")

        # Ensure contiguous memory layout
        sequence_output = sequence_output.squeeze(1).contiguous()

        # Normalize the sequence output
        sequence_output = F.normalize(sequence_output, p=2, dim=-1, eps=1e-12)
        return sequence_output

    def get_video_embeddings(
        self,
        visual_output: torch.Tensor,
        video_mask: torch.Tensor,
        shaped: bool = False,
    ) -> torch.Tensor:
        """Returns L2-normalized video embeddings."""
        if not shaped:
            video_mask = video_mask.reshape(-1, video_mask.shape[-1])

        if not (self.sim_type == "seqTransf" and self.temporal_type == "TDB"):
            raise NotImplementedError(
                f"Unsupported sim_type={self.sim_type!r} or temporal_type={self.temporal_type!r}"
            )

        # Temporal Diffusion Block
        visual_output = visual_output.contiguous()
        visual_output_original = visual_output
        (
            visual_output,
            frame_positional_embeddings,
            type_embedding,
            temporal_video_mask,
        ) = self.temporal_difference_block(visual_output, video_mask)

        # Add frame positional embeddings and type embedding
        visual_output = visual_output + frame_positional_embeddings + type_embedding

        extended_video_mask = (1.0 - temporal_video_mask.unsqueeze(1)) * NEG_INF
        extended_video_mask = extended_video_mask.expand(
            -1, temporal_video_mask.size(1), -1
        )
        visual_output = self.transformerClip(
            visual_output.transpose(0, 1), extended_video_mask
        ).transpose(0, 1)

        # Select even frames
        frame_position_id = torch.arange(
            start=0,
            end=visual_output.size()[1],
            step=2,
            dtype=torch.long,
            device=visual_output.device,
        )
        visual_output = visual_output[:, frame_position_id, :] + visual_output_original
        visual_output = F.normalize(visual_output, p=2, dim=-1, eps=1e-12)

        # Masked mean pooling
        mask_f = video_mask.float().unsqueeze(-1)
        visual_output = (visual_output * mask_f).sum(dim=1)

        denominator = torch.clamp(mask_f.sum(dim=1), min=1.0)
        visual_output = visual_output / denominator

        # Normalize the output
        visual_output = F.normalize(visual_output, p=2, dim=-1, eps=1e-12)

        return visual_output


if __name__ == "__main__":
    pass
