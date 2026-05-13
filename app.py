import streamlit as st
import pandas as pd
import io
from datetime import datetime
from difflib import SequenceMatcher
import plotly.express as px
import plotly.graph_objects as go

# 设置页面布局和标题
st.set_page_config(
    page_title="智慧月台摄像机出入车准确率分析工具",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
    }
    .step {
        text-align: center;
        flex: 1;
        padding: 1rem;
        background-color: #e9ecef;
        border-radius: 5px;
        margin: 0 0.5rem;
    }
    .step.active {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .step.completed {
        background-color: #28a745;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<div class="main-header">🚗 智慧月台摄像机出入车准确率分析工具</div>', unsafe_allow_html=True)
st.markdown("---")

# 初始化会话状态
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'cleaning_done' not in st.session_state:
    st.session_state.cleaning_done = False
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'cleaning_log' not in st.session_state:
    st.session_state.cleaning_log = []
if 'original_count' not in st.session_state:
    st.session_state.original_count = 0
if 'removed_count' not in st.session_state:
    st.session_state.removed_count = 0
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'analyzed_df' not in st.session_state:
    st.session_state.analyzed_df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'plate_analysis_done' not in st.session_state:
    st.session_state.plate_analysis_done = False
if 'plate_analysis_results' not in st.session_state:
    st.session_state.plate_analysis_results = {}
if 'overall_stats' not in st.session_state:
    st.session_state.overall_stats = {}
if 'df' not in st.session_state:
    st.session_state.df = None

# 步骤指示器
def show_step_indicator():
    steps = [
        {"number": 1, "name": "数据上传", "active": st.session_state.current_step == 1},
        {"number": 2, "name": "数据清洗", "active": st.session_state.current_step == 2},
        {"number": 3, "name": "事件检测分析", "active": st.session_state.current_step == 3},
        {"number": 4, "name": "车牌识别分析", "active": st.session_state.current_step == 4},
        {"number": 5, "name": "结果导出", "active": st.session_state.current_step == 5}
    ]
    
    html = '<div class="step-indicator">'
    for step in steps:
        status_class = "active" if step["active"] else "completed" if st.session_state.current_step > step["number"] else ""
        html += f'<div class="step {status_class}">步骤{step["number"]}: {step["name"]}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# 工具函数
def count_character_differences(str1, str2):
    """计算两个字符串的字符差异数量"""
    str1, str2 = str(str1), str(str2)
    max_len = max(len(str1), len(str2))
    str1 = str1.ljust(max_len)
    str2 = str2.ljust(max_len)
    differences = 0
    for i in range(max_len):
        if i < len(str1) and i < len(str2):
            if str1[i] != str2[i]:
                differences += 1
        else:
            differences += 1
    return differences

def analyze_plate_recognition(analyzed_df, event_analysis_results):
    """分析车牌识别情况"""
    plate_results = {}
    
    grouped = analyzed_df.groupby('泊车位')
    
    for parking_space, group in grouped:
        if len(group) < 2:
            continue
            
        plate_results[parking_space] = {
            '总车辆数': 0,
            '车牌识别正常': 0,
            '车牌模糊匹配': 0,
            '车牌识别异常': 0,
            '无车牌': 0,
            '车牌识别准确率': 0.0
        }
        
        group = group.sort_values('序号')
        normal_records = []
        i = 0
        
        while i < len(group) - 1:
            current_record = group.iloc[i]
            next_record = group.iloc[i + 1]
            
            current_idx = current_record.name
            next_idx = next_record.name
            
            current_is_abnormal = analyzed_df.loc[current_idx, '数据分析'] == '出入车事件检测异常'
            
            if current_is_abnormal:
                i += 1
                continue
            
            if (current_record['车辆状态'] == '车辆驶入' and 
                next_record['车辆状态'] == '车辆驶出'):
                normal_records.extend([current_idx, next_idx])
                i += 2
            else:
                if analyzed_df.loc[current_idx, '数据分析'] == '':
                    analyzed_df.loc[current_idx, '数据分析'] = '出入车事件检测异常'
                i += 1
        
        if i == len(group) - 1:
            last_record = group.iloc[i]
            last_idx = last_record.name
            last_is_abnormal = analyzed_df.loc[last_idx, '数据分析'] == '出入车事件检测异常'
            if not last_is_abnormal and last_idx not in normal_records:
                analyzed_df.loc[last_idx, '数据分析'] = '出入车事件检测异常'
        
        normal_group = analyzed_df.loc[normal_records].sort_values('序号')
        
        if len(normal_group) < 2:
            continue
        
        j = 0
        while j < len(normal_group) - 1:
            current_record = normal_group.iloc[j]
            next_record = normal_group.iloc[j + 1]
            
            current_idx = current_record.name
            next_idx = next_record.name
            
            if (current_record['车辆状态'] == '车辆驶入' and 
                next_record['车辆状态'] == '车辆驶出'):
                
                plate_results[parking_space]['总车辆数'] += 1
                current_plate = current_record['车牌号码']
                next_plate = next_record['车牌号码']
                
                if current_plate == '无车牌' or next_plate == '无车牌':
                    plate_results[parking_space]['无车牌'] += 1
                    analyzed_df.loc[current_idx, '数据分析'] = '车牌识别无车牌'
                    analyzed_df.loc[next_idx, '数据分析'] = '车牌识别无车牌'
                else:
                    if current_plate == next_plate:
                        plate_results[parking_space]['车牌识别正常'] += 1
                    else:
                        diff_count = count_character_differences(current_plate, next_plate)
                        if diff_count <= 2:
                            plate_results[parking_space]['车牌模糊匹配'] += 1
                            analyzed_df.loc[current_idx, '数据分析'] = '车牌模糊匹配'
                            analyzed_df.loc[next_idx, '数据分析'] = '车牌模糊匹配'
                        else:
                            plate_results[parking_space]['车牌识别异常'] += 1
                            analyzed_df.loc[current_idx, '数据分析'] = '车牌识别异常'
                            analyzed_df.loc[next_idx, '数据分析'] = '车牌识别异常'
                
                j += 2
            else:
                analyzed_df.loc[current_idx, '数据分析'] = '出入车事件检测异常'
                j += 1
        
        total_plates = plate_results[parking_space]['总车辆数']
        if total_plates > 0:
            normal_plates = plate_results[parking_space]['车牌识别正常']
            plate_results[parking_space]['车牌识别准确率'] = (normal_plates / total_plates * 100)
    
    return analyzed_df, plate_results

def calculate_overall_statistics(event_results, plate_results):
    """计算总体统计信息"""
    total_normal_events = sum(stats['正常事件数'] for stats in event_results.values())
    total_abnormal_events = sum(stats['异常事件数'] for stats in event_results.values())
    total_events = total_normal_events + total_abnormal_events
    
    event_accuracy = total_normal_events / total_events * 100 if total_events > 0 else 0.0
    
    total_plate_vehicles = sum(stats['总车辆数'] for stats in plate_results.values())
    total_plate_normal = sum(stats['车牌识别正常'] for stats in plate_results.values())
    plate_accuracy = total_plate_normal / total_plate_vehicles * 100 if total_plate_vehicles > 0 else 0.0
    
    return {
        '事件检测准确率(%)': f"{event_accuracy:.2f}%",
        '车牌识别准确率(%)': f"{plate_accuracy:.2f}%",
        '总事件数': total_events,
        '正常事件数': total_normal_events,
        '异常事件数': total_abnormal_events,
        '总车辆数': total_plate_vehicles,
        '车牌识别正常': total_plate_normal
    }

def create_visualizations(event_results, plate_results):
    """创建数据可视化图表"""
    # 事件检测准确率图表
    event_data = []
    for parking_space, stats in event_results.items():
        event_data.append({
            '泊车位': parking_space,
            '准确率': stats['准确率'],
            '正常事件数': stats['正常事件数'],
            '异常事件数': stats['异常事件数']
        })
    
    if event_data:
        event_df = pd.DataFrame(event_data)
        fig1 = px.bar(event_df, x='泊车位', y='准确率', 
                      title='各泊车位事件检测准确率',
                      color='准确率', color_continuous_scale='Viridis')
        fig1.update_layout(height=400)
    else:
        fig1 = go.Figure()
        fig1.update_layout(title='无数据可显示', height=400)
    
    # 车牌识别统计图表
    plate_data = []
    for parking_space, stats in plate_results.items():
        plate_data.append({
            '泊车位': parking_space,
            '车牌识别正常': stats['车牌识别正常'],
            '车牌模糊匹配': stats['车牌模糊匹配'],
            '车牌识别异常': stats['车牌识别异常'],
            '无车牌': stats['无车牌']
        })
    
    if plate_data:
        plate_df = pd.DataFrame(plate_data)
        fig2 = px.bar(plate_df, x='泊车位', 
                      y=['车牌识别正常', '车牌模糊匹配', '车牌识别异常', '无车牌'],
                      title='各泊车位车牌识别情况',
                      barmode='stack')
        fig2.update_layout(height=400)
    else:
        fig2 = go.Figure()
        fig2.update_layout(title='无数据可显示', height=400)
    
    return fig1, fig2

# 主程序逻辑
show_step_indicator()

# 步骤1: 数据上传
if st.session_state.current_step >= 1:
    st.markdown('<div class="section-header">📁 步骤1: 数据上传</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("上传Excel文件", type=['xlsx', 'xls'], 
                                   help="请上传包含车辆进出记录的Excel文件",
                                   key="file_uploader")
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            required_columns = ['序号', '泊车位', '车牌号码', '车辆状态']
            
            if not all(col in df.columns for col in required_columns):
                st.error(f"❌ 文件格式错误！需要的列：{required_columns}")
                st.stop()
            
            st.success(f"✅ 文件上传成功！共 {len(df)} 条记录")
            
            with st.expander("📊 数据预览", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                st.write(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 开始数据清洗", type="primary", use_container_width=True, key="start_cleaning"):
                    st.session_state.current_step = 2
                    st.session_state.df = df
                    st.rerun()
            
            with col2:
                if st.button("🔄 重新上传", use_container_width=True, key="reupload_step1"):
                    st.session_state.current_step = 1
                    st.session_state.cleaning_done = False
                    st.session_state.analysis_done = False
                    st.session_state.plate_analysis_done = False
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ 文件读取错误: {str(e)}")

# 步骤2: 数据清洗
if st.session_state.current_step >= 2 and st.session_state.df is not None:
    st.markdown('<div class="section-header">🧹 步骤2: 数据清洗</div>', unsafe_allow_html=True)
    
    if not st.session_state.cleaning_done:
        with st.spinner("正在进行数据清洗..."):
            cleaning_log = []
            cleaned_df = st.session_state.df.copy()
            rows_to_remove = []
            
            grouped = cleaned_df.groupby('泊车位')
            
            for parking_space, group in grouped:
                if len(group) == 0:
                    continue
                    
                cleaning_log.append(f"泊车位 '{parking_space}' 共有 {len(group)} 条记录")
                
                # 检查第一条记录
                first_record = group.iloc[0]
                if first_record['车辆状态'] != '车辆驶入':
                    rows_to_remove.append(first_record.name)
                    cleaning_log.append(f"  - 删除第一条记录（序号 {first_record['序号']}）")
                
                # 检查最后一条记录
                last_record = group.iloc[-1]
                if last_record['车辆状态'] != '车辆驶出':
                    if len(group) >= 2:
                        second_last_record = group.iloc[-2]
                        if second_last_record['车辆状态'] == '车辆驶入' and last_record['车辆状态'] == '车辆驶入':
                            rows_to_remove.append(last_record.name)
                            cleaning_log.append(f"  - 删除最后一条记录（序号 {last_record['序号']}）")
                        else:
                            rows_to_remove.append(last_record.name)
                            cleaning_log.append(f"  - 删除最后一条记录（序号 {last_record['序号']}）")
                    else:
                        rows_to_remove.append(last_record.name)
                        cleaning_log.append(f"  - 删除最后一条记录（序号 {last_record['序号']}）")
            
            original_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop(rows_to_remove)
            removed_count = original_count - len(cleaned_df)
            
            st.session_state.cleaning_done = True
            st.session_state.cleaned_df = cleaned_df
            st.session_state.cleaning_log = cleaning_log
            st.session_state.original_count = original_count
            st.session_state.removed_count = removed_count
    
    if st.session_state.cleaning_done:
        st.markdown('<div class="success-box">✅ 数据清洗完成</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("原始记录数", st.session_state.original_count)
        with col2:
            st.metric("删除记录数", st.session_state.removed_count, delta=-st.session_state.removed_count)
        with col3:
            st.metric("剩余记录数", len(st.session_state.cleaned_df))
        
        with st.expander("📋 清洗详情", expanded=True):
            tab1, tab2 = st.tabs(["清洗后数据", "清洗日志"])
            with tab1:
                st.dataframe(st.session_state.cleaned_df, use_container_width=True, height=300)
            with tab2:
                for log_entry in st.session_state.cleaning_log:
                    st.write(f"• {log_entry}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 开始事件检测分析", type="primary", use_container_width=True, key="start_event_analysis"):
                st.session_state.current_step = 3
                st.rerun()
        with col2:
            if st.button("↩️ 返回上一步", use_container_width=True, key="back_to_step1"):
                st.session_state.current_step = 1
                st.rerun()

# 步骤3: 事件检测分析
if st.session_state.current_step >= 3 and st.session_state.cleaning_done:
    st.markdown('<div class="section-header">🔍 步骤3: 事件检测分析</div>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_done:
        with st.spinner("正在进行事件检测分析..."):
            analyzed_df = st.session_state.cleaned_df.copy()
            analysis_results = {}
            
            if len(analyzed_df.columns) >= 6:
                analyzed_df.insert(5, '数据分析', '')
            else:
                analyzed_df['数据分析'] = ''
            
            grouped = analyzed_df.groupby('泊车位')
            
            for parking_space, group in grouped:
                if len(group) < 2:
                    continue
                    
                analysis_results[parking_space] = {
                    '总记录数': len(group),
                    '正常事件数': 0,
                    '异常事件数': 0,
                    '准确率': 0.0,
                    '总车辆数': 0
                }
                
                i = 0
                processed_indices = set()
                
                while i < len(group) - 1:
                    current_record = group.iloc[i]
                    next_record = group.iloc[i + 1]
                    
                    current_idx = current_record.name
                    next_idx = next_record.name
                    
                    if current_idx in processed_indices:
                        i += 1
                        continue
                    
                    if (current_record['车辆状态'] == '车辆驶入' and 
                        next_record['车辆状态'] == '车辆驶出'):
                        analysis_results[parking_space]['正常事件数'] += 1
                        analysis_results[parking_space]['总车辆数'] += 1
                        processed_indices.add(current_idx)
                        processed_indices.add(next_idx)
                        i += 2
                    else:
                        analyzed_df.loc[current_idx, '数据分析'] = '出入车事件检测异常'
                        analysis_results[parking_space]['异常事件数'] += 1
                        analysis_results[parking_space]['总车辆数'] += 1
                        processed_indices.add(current_idx)
                        i += 1
                
                if i == len(group) - 1 and i not in processed_indices:
                    last_record = group.iloc[i]
                    last_idx = last_record.name
                    analyzed_df.loc[last_idx, '数据分析'] = '出入车事件检测异常'
                    analysis_results[parking_space]['异常事件数'] += 1
                    analysis_results[parking_space]['总车辆数'] += 1
                
                total_events = analysis_results[parking_space]['正常事件数'] + analysis_results[parking_space]['异常事件数']
                if total_events > 0:
                    analysis_results[parking_space]['准确率'] = (
                        analysis_results[parking_space]['正常事件数'] / total_events * 100
                    )
            
            st.session_state.analysis_done = True
            st.session_state.analyzed_df = analyzed_df
            st.session_state.analysis_results = analysis_results
    
    if st.session_state.analysis_done:
        st.markdown('<div class="success-box">✅ 事件检测分析完成</div>', unsafe_allow_html=True)
        
        # 总体统计
        total_normal = sum(stats['正常事件数'] for stats in st.session_state.analysis_results.values())
        total_abnormal = sum(stats['异常事件数'] for stats in st.session_state.analysis_results.values())
        total_accuracy = total_normal / (total_normal + total_abnormal) * 100 if (total_normal + total_abnormal) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总记录数", len(st.session_state.analyzed_df))
        with col2:
            st.metric("正常事件数", total_normal)
        with col3:
            st.metric("异常事件数", total_abnormal)
        with col4:
            st.metric("事件检测准确率", f"{total_accuracy:.2f}%")
        
        with st.expander("📈 分析结果详情", expanded=True):
            tab1, tab2 = st.tabs(["分析数据", "统计报表"])
            with tab1:
                st.dataframe(st.session_state.analyzed_df, use_container_width=True, height=300)
            with tab2:
                stats_data = []
                for parking_space, stats in st.session_state.analysis_results.items():
                    stats_data.append({
                        '泊车位': parking_space,
                        '总记录数': stats['总记录数'],
                        '正常事件数': stats['正常事件数'],
                        '异常事件数': stats['异常事件数'],
                        '总车辆数': stats['总车辆数'],
                        '事件检测准确率(%)': f"{stats['准确率']:.2f}%"
                    })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔢 开始车牌识别分析", type="primary", use_container_width=True, key="start_plate_analysis"):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("↩️ 返回上一步", use_container_width=True, key="back_to_step2"):
                st.session_state.current_step = 2
                st.session_state.analysis_done = False
                st.rerun()

# 步骤4: 车牌识别分析
if st.session_state.current_step >= 4 and st.session_state.analysis_done:
    st.markdown('<div class="section-header">🔢 步骤4: 车牌识别分析</div>', unsafe_allow_html=True)
    
    if not st.session_state.plate_analysis_done:
        with st.spinner("正在进行车牌识别分析..."):
            analyzed_df_with_plate, plate_results = analyze_plate_recognition(
                st.session_state.analyzed_df.copy(), 
                st.session_state.analysis_results
            )
            
            overall_stats = calculate_overall_statistics(
                st.session_state.analysis_results,
                plate_results
            )
            
            st.session_state.plate_analysis_done = True
            st.session_state.analyzed_df = analyzed_df_with_plate
            st.session_state.plate_analysis_results = plate_results
            st.session_state.overall_stats = overall_stats
    
    if st.session_state.plate_analysis_done:
        st.markdown('<div class="success-box">✅ 车牌识别分析完成</div>', unsafe_allow_html=True)
        
        # 可视化图表
        try:
            fig1, fig2 = create_visualizations(st.session_state.analysis_results, st.session_state.plate_analysis_results)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
        except:
            st.info("图表生成需要足够的数据支持")
        
        # 总体统计
        total_plate_vehicles = sum(stats['总车辆数'] for stats in st.session_state.plate_analysis_results.values())
        total_plate_normal = sum(stats['车牌识别正常'] for stats in st.session_state.plate_analysis_results.values())
        total_plate_accuracy = total_plate_normal / total_plate_vehicles * 100 if total_plate_vehicles > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总车辆数", total_plate_vehicles)
        with col2:
            st.metric("车牌识别正常", total_plate_normal)
        with col3:
            st.metric("车牌识别准确率", f"{total_plate_accuracy:.2f}%")
        with col4:
            st.metric("事件检测准确率", st.session_state.overall_stats['事件检测准确率(%)'])
        
        with st.expander("📊 详细分析结果", expanded=True):
            tab1, tab2, tab3 = st.tabs(["最终数据", "车牌统计", "异常详情"])
            with tab1:
                st.dataframe(st.session_state.analyzed_df, use_container_width=True, height=300)
            with tab2:
                plate_stats_data = []
                for parking_space, stats in st.session_state.plate_analysis_results.items():
                    event_stats = st.session_state.analysis_results.get(parking_space, {})
                    plate_stats_data.append({
                        '泊车位': parking_space,
                        '总车辆数': stats['总车辆数'],
                        '车牌识别正常': stats['车牌识别正常'],
                        '车牌模糊匹配': stats['车牌模糊匹配'],
                        '车牌识别异常': stats['车牌识别异常'],
                        '无车牌': stats['无车牌'],
                        '车牌识别准确率(%)': f"{stats['车牌识别准确率']:.2f}%"
                    })
                
                if plate_stats_data:
                    plate_stats_df = pd.DataFrame(plate_stats_data)
                    st.dataframe(plate_stats_df, use_container_width=True)
            with tab3:
                abnormal_records = st.session_state.analyzed_df[
                    st.session_state.analyzed_df['数据分析'].isin(['车牌识别异常', '车牌模糊匹配', '车牌识别无车牌', '出入车事件检测异常'])
                ]
                if len(abnormal_records) > 0:
                    st.dataframe(abnormal_records, use_container_width=True, height=300)
                else:
                    st.info("未发现异常记录")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 进入结果导出", type="primary", use_container_width=True, key="go_to_export"):
                st.session_state.current_step = 5
                st.rerun()
        with col2:
            if st.button("↩️ 返回上一步", use_container_width=True, key="back_to_step3"):
                st.session_state.current_step = 3
                st.session_state.plate_analysis_done = False
                st.rerun()

# 步骤5: 结果导出
if st.session_state.current_step >= 5 and st.session_state.plate_analysis_done:
    st.markdown('<div class="section-header">📥 步骤5: 结果导出</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 分析完成！请选择导出格式并下载结果文件</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("导出设置")
        export_format = st.selectbox("导出格式", ["Excel", "CSV"], key="export_format")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = st.text_input("文件名", value=f"车辆分析报告_{timestamp}", key="export_filename")
        
        if export_format == "Excel":
            st.info("Excel格式将包含多个工作表：分析结果、统计报表、清洗日志等")
        else:
            st.info("CSV格式仅导出分析后的数据表格")
    
    with col2:
        st.subheader("导出内容预览")
        
        # 总体统计卡片
        st.markdown("### 📊 总体统计")
        overall_col1, overall_col2 = st.columns(2)
        with overall_col1:
            st.metric("事件检测准确率", st.session_state.overall_stats['事件检测准确率(%)'])
        with overall_col2:
            st.metric("车牌识别准确率", st.session_state.overall_stats['车牌识别准确率(%)'])
        
        # 数据预览
        with st.expander("📋 数据预览", expanded=True):
            st.dataframe(st.session_state.analyzed_df.head(10), use_container_width=True)
    
    # 导出按钮
    if export_format == "Excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state.cleaned_df.to_excel(writer, index=False, sheet_name='清洗结果')
            st.session_state.analyzed_df.to_excel(writer, index=False, sheet_name='分析结果')
            
            log_df = pd.DataFrame(st.session_state.cleaning_log, columns=['清洗日志'])
            log_df.to_excel(writer, index=False, sheet_name='清洗日志')
            
            # 事件检测统计
            stats_data = []
            for parking_space, stats in st.session_state.analysis_results.items():
                stats_data.append({
                    '泊车位': parking_space,
                    '总记录数': stats['总记录数'],
                    '正常事件数': stats['正常事件数'],
                    '异常事件数': stats['异常事件数'],
                    '总车辆数': stats['总车辆数'],
                    '事件检测准确率(%)': f"{stats['准确率']:.2f}%"
                })
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, index=False, sheet_name='事件检测统计')
            
            # 车牌识别统计
            plate_stats_data = []
            for parking_space, stats in st.session_state.plate_analysis_results.items():
                plate_stats_data.append({
                    '泊车位': parking_space,
                    '总车辆数': stats['总车辆数'],
                    '车牌识别正常': stats['车牌识别正常'],
                    '车牌模糊匹配': stats['车牌模糊匹配'],
                    '车牌识别异常': stats['车牌识别异常'],
                    '无车牌': stats['无车牌'],
                    '车牌识别准确率(%)': f"{stats['车牌识别准确率']:.2f}%"
                })
            if plate_stats_data:
                plate_stats_df = pd.DataFrame(plate_stats_data)
                plate_stats_df.to_excel(writer, index=False, sheet_name='车牌识别统计')
            
            # 总体统计
            overall_stats_df = pd.DataFrame([st.session_state.overall_stats])
            overall_stats_df.to_excel(writer, index=False, sheet_name='总体统计')
        
        excel_data = output.getvalue()
        
        st.download_button(
            label="📥 下载Excel文件",
            data=excel_data,
            file_name=f"{export_filename}.xlsx",
            mime="application/vnd.ms-excel",
            use_container_width=True,
            type="primary",
            key="download_excel"
        )
    
    else:  # CSV
        csv_data = st.session_state.analyzed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载CSV文件",
            data=csv_data,
            file_name=f"{export_filename}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
            key="download_csv"
        )
    
    st.markdown("---")
    if st.button("🔄 开始新的分析", use_container_width=True, key="start_new_analysis"):
        # 重置所有状态
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.current_step = 1
        st.rerun()

# 侧边栏信息
with st.sidebar:
    st.header("ℹ️ 使用说明")
    st.markdown("""
    **分析流程：**
    1. **上传数据** - 支持.xlsx/.xls格式
    2. **数据清洗** - 自动清理异常记录
    3. **事件检测** - 分析车辆进出事件
    4. **车牌识别** - 评估车牌识别准确性
    5. **结果导出** - 下载分析报告
    
    **数据要求：**
    - 必须包含列：序号、泊车位、车牌号码、车辆状态
    - 车辆状态应为：车辆驶入/车辆驶出
    """)
    
    if st.session_state.plate_analysis_done:
        st.header("📈 分析摘要")
        st.metric("总记录数", st.session_state.original_count)
        st.metric("有效记录数", len(st.session_state.analyzed_df))
        st.metric("事件检测准确率", st.session_state.overall_stats['事件检测准确率(%)'])
        st.metric("车牌识别准确率", st.session_state.overall_stats['车牌识别准确率(%)'])
