"""自动保存能力测试日志，以及问题和回答。"""

import contextlib
from datetime import datetime
import functools
import json
from pathlib import Path
import sys
import traceback


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, text):
        self.logfile.write(text)
        self.logfile.flush()
        return self.terminal.write(text)

    def flush(self):
        self.logfile.flush()
        self.terminal.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def run_logged(main):
    namespace = main.__globals__
    root = Path(namespace["__file__"]).resolve().parent

    folder = (
        root
        / "test_logs"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    folder.mkdir(parents=True, exist_ok=False)

    records = []
    original_ask = namespace["ask"]

    def save_answers():
        sections = []

        for record in records:
            section = (
                f"问题{record['number']}：{record['question']}\n"
                f"回答：{record.get('answer', '')}"
            )

            if "error" in record:
                section += f"\n执行错误：{record['error']}"

            sections.append(section)

        (folder / "questions_answers.txt").write_text(
            "\n\n".join(sections) + "\n",
            encoding="utf-8-sig",
        )

        (folder / "questions_answers.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @functools.wraps(original_ask)
    def recorded_ask(
        agent,
        number,
        category,
        question,
        image_path=None,
        video_path=None,
    ):
        record = {
            "number": number,
            "category": category,
            "question": question,
            "image_path": image_path,
            "video_path": video_path,
        }

        try:
            answer = original_ask(
                agent,
                number,
                category,
                question,
                image_path=image_path,
                video_path=video_path,
            )
            record["answer"] = answer
            return answer

        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

        finally:
            records.append(record)
            save_answers()

    namespace["ask"] = recorded_ask

    try:
        with (folder / "run.log").open(
            "w", encoding="utf-8-sig"
        ) as logfile:
            with (
                contextlib.redirect_stdout(Tee(sys.stdout, logfile)),
                contextlib.redirect_stderr(Tee(sys.stderr, logfile)),
            ):
                print(f"本次测试输出目录：{folder}")
                status = "运行中"

                try:
                    main()
                    status = "执行完成（回答正确性需另行核对）"

                except BaseException:
                    status = "执行中断或失败"
                    traceback.print_exc()
                    raise

                finally:
                    save_answers()

                    (folder / "status.txt").write_text(
                        status + "\n",
                        encoding="utf-8-sig",
                    )

                    print(f"测试状态：{status}")
                    print(f"日志和问答已保存至：{folder}")

    finally:
        namespace["ask"] = original_ask