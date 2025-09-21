#!/usr/bin/env python3
"""
Summary service to compute quick rundown stats and craft a speech-friendly script.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class SummaryService:
    def __init__(self):
        pass

    @staticmethod
    def _parse_event_time(ts: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None

    @staticmethod
    def _fmt_time(dt: datetime) -> str:
        try:
            return dt.strftime('%-I:%M %p')
        except ValueError:
            # Windows does not support '-' flag; fallback
            return dt.strftime('%I:%M %p').lstrip('0')

    def compute_stats(self, events: List[Dict], hours: int = 24) -> Dict:
        now = datetime.now()
        cutoff = now.timestamp() - hours * 3600

        motion_times: List[datetime] = []
        fall_times: List[datetime] = []

        earliest: Optional[datetime] = None
        latest: Optional[datetime] = None

        for e in events:
            dt = self._parse_event_time(e.get('timestamp', ''))
            if not dt:
                continue
            if dt.timestamp() < cutoff:
                continue

            if earliest is None or dt < earliest:
                earliest = dt
            if latest is None or dt > latest:
                latest = dt

            class_name = (e.get('class_name') or '').lower()
            category = (e.get('metadata', {}).get('category') or '').lower()

            if 'motion' in class_name or category == 'motion':
                motion_times.append(dt)
            if 'fall' in class_name or ('activity' == category and 'fall' in class_name):
                fall_times.append(dt)

        motion_times.sort()
        fall_times.sort()

        return {
            'motion_count': len(motion_times),
            'fall_count': len(fall_times),
            'fall_times': fall_times[-3:],  # last up to 3
            'since_time': earliest,
            'latest_time': latest,
            'hours_window': hours
        }

    def next_medication(self, meds: List[Dict]) -> Optional[Tuple[str, datetime]]:
        now = datetime.now()
        upcoming: List[Tuple[str, datetime]] = []
        for med in meds or []:
            times_str = med.get('times') or ''
            name = med.get('name') or 'Medication'
            for t in [t.strip() for t in times_str.split(',') if t.strip()]:
                try:
                    rt = datetime.strptime(t, '%I:%M %p').time()
                    dt = now.replace(hour=rt.hour, minute=rt.minute, second=0, microsecond=0)
                    if dt > now:
                        upcoming.append((name, dt))
                except Exception:
                    continue
        if not upcoming:
            return None
        upcoming.sort(key=lambda x: x[1])
        return upcoming[0]

    def craft_rundown(self, events: List[Dict], meds: List[Dict]) -> str:
        stats = self.compute_stats(events, hours=24)
        since = stats['since_time']
        motion_count = stats['motion_count']
        fall_count = stats['fall_count']
        fall_times = stats['fall_times']

        parts: List[str] = []

        if since:
            parts.append(f"Since {self._fmt_time(since)}")
        else:
            parts.append("In the last day")

        # Core counts
        counts_clause = []
        counts_clause.append(f"{motion_count} motion event{'s' if motion_count != 1 else ''}")
        counts_clause.append(f"{fall_count} fall{'s' if fall_count != 1 else ''}")
        parts.append(": ")
        parts.append(", ".join(counts_clause))

        # Fall times brief
        if fall_count > 0 and fall_times:
            times_str = " and ".join([self._fmt_time(t) for t in fall_times[-2:]])
            parts.append(f"; latest falls at {times_str}")

        # Upcoming medication
        next_med = self.next_medication(meds)
        if next_med:
            name, dt = next_med
            parts.append(f". Next medication: {name} at {self._fmt_time(dt)}")

        sentence = "".join(parts).strip()
        if not sentence.endswith('.'):
            sentence += '.'
        return sentence


