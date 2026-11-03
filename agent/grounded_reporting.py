"""Deterministic reporting of observed labels, sources and sample scores."""
import os

VERBS = {
    'grasp': '抓持', 'retract': '牵拉', 'dissect': '分离',
    'coagulate': '电凝', 'cut': '切割', 'clip': '夹闭',
    'aspirate': '吸引', 'irrigate': '冲洗', 'pack': '填塞',
}


def needs_explanation(question):
    return any(word in question.lower() for word in (
        '知识', '图谱', 'class', 'cholect', '含义', '意思', '解释',
        '为什么', '还能', '用途', '是否可以认为',
    ))


def image_scope(question, image_path=None):
    if image_path:
        return True
    return any(word in question for word in ('图片', '图像', 'test.jpg', '刚才识别'))


def graph_only(question):
    text = question.lower()
    return any(w in text for w in ('图谱', 'cholect', 'class')) and not any(
        w in text for w in ('视频', '秒', '图像', '图片', '这个', '刚才', '上述')
    )


def render_event(event, sample=None):
    if not event:
        return '未检索到事件。'
    sample = sample or event.get('query_sample')
    start, end = event.get('start_time'), event.get('end_time')
    lines = [f'采样事件范围：{start}～{end}秒']
    if sample:
        lines.append(f"采样点：{sample.get('timestamp')}秒；帧号：{sample.get('frame_id')}")
    if event.get('type') == 'no_action':
        lines.append('未识别到完整有效动作标签；不能据此断言没有操作。')
        return '\n'.join(lines)
    confidence = (sample or {}).get('confidence', event.get('mean_confidence'))
    label = '—'.join(str(event.get(k, '未知')) for k in ('instrument', 'verb', 'target'))
    verb = VERBS.get(event.get('verb'), event.get('verb', '未知'))
    if confidence is None:
        policy = '置信度缺失，不能作确定判断'
    elif confidence < 0.5 or event.get('type') == 'uncertain':
        policy = 'low：仅为候选，不能作确定判断'
    elif confidence < 0.75:
        policy = 'medium：可能为该操作，建议复核'
    else:
        policy = 'high：模型识别结果'
    lines.extend([f'三元组：{label}', f'动作译名：{verb}',
                  f"{'采样点置信度' if sample else '事件平均置信度'}：{confidence}", policy])
    return '\n'.join(lines)


def render_video(events, video_path):
    source = os.path.basename(video_path) if video_path else '来源未记录'
    lines = [f'【视频识别结果】来源：{source}']
    lines.extend(f'{i}. {render_event(event)}' for i, event in enumerate(events, 1))
    if not events:
        lines.append('没有可用事件。')
    lines.append('范围为采样事件跨度，不是逐帧确认的动作边界；采样间隙不作推断。')
    return '\n\n'.join(lines)
