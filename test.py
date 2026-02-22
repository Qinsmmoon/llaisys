import os
import subprocess
import tempfile

def create_silence_mp3(duration_seconds=10, output_path=None):
    """创建一个指定时长的静音MP3文件"""
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"silence_{duration_seconds}s.mp3")
    
    try:
        # 使用ffmpeg创建静音MP3（需要安装ffmpeg）
        # 如果系统没有ffmpeg，会回退到原来的方法
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo',
            '-t', str(duration_seconds),
            '-c:a', 'libmp3lame',
            '-b:a', '128k',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ 未找到ffmpeg，使用简单的静音片段（可能在某些播放器不兼容）")
        return None

def merge_mp3_from_folder():
    """从指定文件夹合并MP3文件，文件间有10秒静音间隔"""
    
    # 文件夹路径
    folder_path = r"C:\Users\14259\Desktop\新建文件夹"
    
    # 按照你需要的顺序列出文件名
    mp3_files = [
        "1.转肩《禅境》.mp3",
        "2.热身Сен Маған Массың - Рахымжан.mp3",
        "3.S腰（那就跳舞） (DJR7版) - 旺姆.mp3",
        "4.瘦腿瘦腰（DA DA DA ）- 刘至佳.mp3",
        "5.全身燃脂（最炫民族风）  - GoLin.mp3",
        "6.全身运动 - 漫步人生路 (粤语 Electro Rmx) - Dj.mp3",
        "7.颈部《姐就是女王》.mp3",
        "8.颈部画v.mp3",
        "9.开肩《落差》.mp3",
        "10.肩部练习.m4a",
        "11.肩部上手转肩《 风月》.mp3",
        "12.超模腿《忘川彼岸》.mp3",
        "13.甩臂音乐.mp3",
        "14.背部《黑桃A 》.mp3",
        "15.背部高位下拉.mp3",
        "16.回春术。护花使者.mp3",
        "17.美胸《一笑江湖》.mp3",
        "18.魅力腰胯《小蛮腰》.mp3",
        "19.前后动跨.mp3",
        "20.手臂《卜挂》.mp3",
        "21.登山步。一生有你.mp3",
        "22.力量Silver Smoke_重复4次.mp3",
        "23.我们都老了 - 马健涛_重复2次.mp3"
    ]
    
    output_file = os.path.join(folder_path, "健身操音乐合集_5秒间隔.mp3")
    pause_seconds = 5
    
    print("=" * 60)
    print("健身操音乐合并工具")
    print("=" * 60)
    print(f"文件夹路径: {folder_path}")
    print(f"文件间间隔: {pause_seconds}秒")
    print(f"输出文件: {output_file}")
    print("-" * 60)
    
    # 检查文件是否存在
    existing_files = []
    missing_files = []
    
    for filename in mp3_files:
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024
            existing_files.append((file_path, filename, file_size))
            print(f"✅ {filename} ({file_size:.2f} KB)")
        else:
            missing_files.append(filename)
            print(f"❌ {filename} (不存在)")
    
    if missing_files:
        print("\n⚠️ 以下文件不存在：")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请检查文件名是否完全一致")
        return
    
    print(f"\n找到 {len(existing_files)} 个文件，开始合并...")
    
    # 尝试创建静音文件
    silence_file = create_silence_mp3(pause_seconds)
    use_simple_silence = silence_file is None
    
    try:
        with open(output_file, 'wb') as outfile:
            total_files = len(existing_files)
            total_size = 0
            
            for idx, (file_path, filename, file_size) in enumerate(existing_files, 1):
                print(f"\r处理中: {idx}/{total_files} - {filename}", end="")
                
                # 读取并写入当前文件
                with open(file_path, 'rb') as infile:
                    data = infile.read()
                    outfile.write(data)
                    total_size += len(data)
                
                # 如果不是最后一个文件，添加静音
                if idx < total_files:
                    if use_simple_silence:
                        # 简单的静音数据
                        silence_data = b'\x00' * (44100 * 2 * 2 * pause_seconds)
                        outfile.write(silence_data)
                        total_size += len(silence_data)
                    else:
                        # 读取并写入静音文件
                        with open(silence_file, 'rb') as silfile:
                            silence_data = silfile.read()
                            outfile.write(silence_data)
                            total_size += len(silence_data)
            
            final_size_mb = total_size / (1024 * 1024)
            print(f"\n\n✅ 合并成功！")
            print(f"输出文件: {output_file}")
            print(f"文件大小: {final_size_mb:.2f} MB")
            
            # 清理临时文件
            if not use_simple_silence and os.path.exists(silence_file):
                os.unlink(silence_file)
            
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")

def list_files_in_folder():
    """列出文件夹中的所有音频文件"""
    folder_path = r"C:\Users\14259\Desktop\新建文件夹"
    
    print(f"文件夹: {folder_path}")
    print("-" * 40)
    
    # 列出所有音频文件
    audio_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.mp3', '.m4a', '.wav', '.flac')):
            file_path = os.path.join(folder_path, file)
            file_size = os.path.getsize(file_path) / 1024
            audio_files.append((file, file_size))
    
    # 按文件名排序
    audio_files.sort(key=lambda x: x[0])
    
    for file, size in audio_files:
        print(f"{file} ({size:.2f} KB)")
    
    return audio_files

if __name__ == "__main__":
    # 如果你想先查看文件夹中的文件，取消下面的注释
    # list_files_in_folder()
    
    merge_mp3_from_folder()