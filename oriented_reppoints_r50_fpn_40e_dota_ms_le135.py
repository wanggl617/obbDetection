_base_ = ['./oriented_reppoints_r50_fpn_1x_dota_le135.py']
angle_version = 'le135'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RResize',
        img_scale=[(1333, 768), (1333, 1280)],
        multiscale_mode='range'),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# evaluation
evaluation = dict(interval=40, metric='mAP')
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 32, 38])
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=2)

'''
python ./tools/test.py \
  configs/oriented_reppoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135.py \
  checkpoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135-bb0323fd.pth \
  --show-dir work_dirs/vis
  python ./tools/test.py  \
  configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py \
  checkpoints/oriented_reppoints_r50_fpn_1x_dota_le135-ef072de9.pth --format-only \
  --eval-options submission_dir=work_dirs/Task1_results

  python ./tools/test.py  \
  configs/oriented_reppoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135.py \
  checkpoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135-bb0323fd.pth --format-only \
  --eval-options submission_dir=work_dirs/Task2_results

python ./tools/test.py work_dirs/fair1m/oriented_reppoints_r50_fpn_1x_dota_le135.py work_dirs/fair1m/epoch_12.pth  --format-only --eval-options submission_dir=work_dirs/fair1m/Task_fair1m_results

  python ./tools/test.py \
  configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py \
  checkpoints/oriented_reppoints_r50_fpn_1x_dota_le135-ef072de9.pth  --eval mAP
  python ./tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_6x_hrsc_rr_oc.py \
  checkpoints/rotated_retinanet_hbb_r50_fpn_6x_hrsc_rr_oc-f37eada6.pth  --eval mAP

  python ./tools/test.py \
  configs/oriented_reppoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135.py \
  checkpoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135-bb0323fd.pth  --eval mAP

  
python tools/train.py \
configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py \
 --no-validate\
 --resume-from work_dir/oriented_reppoints_r50_fpn_1x_dota_le135/latest.pth



python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_trainval_fair1m.json


python3 ./tools/test.py configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py checkpoints/oriented_reppoints_r50_fpn_1x_dota_le135-ef072de9.pth --out=latest.pkl
python tools/analysis_tools/confusion_matrix.py \
configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py \
latest.pkl work_dirs/dotav1 --show
python tools/analysis_tools/confusion_matrix.py work_dirs/fair1m/oriented_reppoints_r50_fpn_1x_dota_le135.py fair1m.pkl work_dirs/fair1m --show

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/dotav1/loss.json --keys s0.loss_cls sr0.loss_cls --legend Softmax_CE SRW_CE --out losses_dota.png
'''
