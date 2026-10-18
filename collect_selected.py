"""List and optionally collect exactly the outputs selected by FaceReclaim."""
import argparse
import json
from pathlib import Path
import shutil

from reclaim_from_clip import selected_paths_with_backfill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fake_root', type=Path, required=True)
    parser.add_argument('--save_root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--collect_root', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_source_ids', type=int, default=5)
    parser.add_argument('--ratio', type=int, default=-5)
    args = parser.parse_args()
    if args.ratio >= -1 or args.max_source_ids < 1:
        parser.error('use --ratio -5 and a positive --max_source_ids')
    source = args.fake_root.resolve()
    output = args.save_root.resolve()
    if not source.is_dir() or not output.is_dir():
        parser.error('fake_root and save_root must be existing directories')
    if args.collect_root:
        destination = args.collect_root.resolve()
        for existing in (source, output):
            if destination == existing or destination in existing.parents or existing in destination.parents:
                parser.error('collect_root must be separate from both input and output')
        if destination.exists():
            parser.error('collect_root must be a new directory to avoid retaining extra images')
    paths, unreadable = selected_paths_with_backfill(source, args)
    names = sorted(p.relative_to(source) for p in paths)
    available = [name for name in names if (output/name).is_file()]
    missing = [name for name in names if not (output/name).is_file()]
    report = dict(fake_root=str(source), save_root=str(output), seed=args.seed,
                  max_source_ids=args.max_source_ids, ratio=args.ratio,
                  selected=len(names), existing=len(available), missing=len(missing),
                  selected_paths=[p.as_posix() for p in names],
                  missing_paths=[p.as_posix() for p in missing], unreadable_inputs=unreadable)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Selected: {len(names)}, existing: {len(available)}, missing: {len(missing)}')
    print(f'Manifest: {args.manifest.resolve()}')
    if args.collect_root:
        destination.mkdir(parents=True)
        for name in available:
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output/name, target)
        print(f'Collected {len(available)} existing outputs into {destination}')
        if missing:
            print('Collection is incomplete; restore missing outputs before using it as the final dataset.')


if __name__ == '__main__':
    main()
