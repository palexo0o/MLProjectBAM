"""robots.txt handling with Google-style wildcard support.

urllib.robotparser does not understand the ``*`` and ``$`` wildcards that
Idealista uses, so we implement the matching ourselves. Only the ``*`` user-agent
group is honoured, since this scraper does not identify as a named crawler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass
class _Rule:
    allow: bool
    pattern: re.Pattern[str]
    length: int


@dataclass
class RobotsRules:
    rules: list[_Rule] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> "RobotsRules":
        rules: list[_Rule] = []
        in_star_group = False
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                in_star_group = value == "*"
            elif in_star_group and key in ("allow", "disallow") and value:
                rules.append(
                    _Rule(
                        allow=key == "allow",
                        pattern=_compile(value),
                        length=len(value),
                    )
                )
        return cls(rules)

    def can_fetch(self, url: str) -> bool:
        """Return True if the path+query of ``url`` may be fetched.

        The longest matching rule wins; on a tie, Allow beats Disallow.
        """
        parts = urlsplit(url)
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query

        best: _Rule | None = None
        for rule in self.rules:
            if rule.pattern.match(target):
                if best is None or rule.length > best.length or (
                    rule.length == best.length and rule.allow
                ):
                    best = rule
        return best is None or best.allow


def _compile(pattern: str) -> re.Pattern[str]:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = ".*".join(re.escape(chunk) for chunk in body.split("*"))
    return re.compile("^" + regex + ("$" if anchored else ""))
