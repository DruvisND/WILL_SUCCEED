# GitHub 使用指南

本指南帮助您快速掌握GitHub的核心功能和工作流程。

## 目录

1. [GitHub基础概念](#1-github基础概念)
2. [Git安装与配置](#2-git安装与配置)
3. [仓库操作](#3-仓库操作)
4. [代码管理](#4-代码管理)
5. [分支操作](#5-分支操作)
6. [协作开发](#6-协作开发)
7. [GitHub功能](#7-github功能)
8. [常见问题](#8-常见问题)

---

## 1. GitHub基础概念

### 1.1 什么是Git和GitHub？

| 概念 | 说明 |
|------|------|
| **Git** | 分布式版本控制系统，用于跟踪代码变更 |
| **GitHub** | 基于Git的代码托管平台，提供远程仓库服务 |
| **Repository（仓库）** | 存储项目代码和历史的地方 |
| **Commit（提交）** | 保存代码的某个版本快照 |
| **Branch（分支）** | 独立开发线，允许并行工作 |
| **Merge（合并）** | 将不同分支的代码合并到一起 |
| **Pull Request（拉取请求）** | 请求将代码合并到主分支 |

### 1.2 常见术语

```
origin     - 默认远程仓库别名
master/main - 主分支名称
HEAD       - 当前所在位置（分支/提交）
stash      - 临时保存未提交的修改
```

---

## 2. Git安装与配置

### 2.1 安装Git（Windows）

```powershell
# 方法1: 使用Chocolatey（推荐）
choco install git

# 方法2: 手动下载
# 访问 https://git-scm.com/download/win 下载安装
```

### 2.2 初始配置

```bash
# 设置用户名（必填）
git config --global user.name "Your Name"

# 设置邮箱（必填）
git config --global user.email "your.email@example.com"

# 设置默认分支名
git config --global init.defaultBranch main

# 设置合并工具（可选）
git config --global merge.tool vscode

# 查看所有配置
git config --list

# 查看特定配置
git config user.name
```

### 2.3 生成SSH密钥（推荐）

```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "your.email@example.com"

# 查看公钥内容
cat ~/.ssh/id_ed25519.pub

# 将公钥添加到GitHub：
# 1. 访问 GitHub → Settings → SSH and GPG keys → New SSH key
# 2. 粘贴公钥内容
```

---

## 3. 仓库操作

### 3.1 创建本地仓库

```bash
# 初始化新仓库
git init

# 克隆远程仓库
git clone https://github.com/username/repository.git
git clone git@github.com:username/repository.git  # SSH方式

# 克隆指定分支
git clone -b develop https://github.com/username/repository.git
```

### 3.2 连接远程仓库

```bash
# 查看远程仓库
git remote -v

# 添加远程仓库
git remote add origin https://github.com/username/repository.git

# 修改远程仓库地址
git remote set-url origin new-url

# 删除远程仓库
git remote remove origin
```

### 3.3 文件操作

```bash
# 查看仓库状态
git status

# 查看文件差异
git diff                    # 工作区 vs 暂存区
git diff --staged           # 暂存区 vs 最新提交
git diff HEAD               # 工作区 vs 最新提交

# 添加文件到暂存区
git add filename.txt        # 单个文件
git add .                   # 所有文件
git add *.py                # 特定类型文件
git add -p                  # 交互式添加

# 移除文件
git rm filename.txt         # 删除文件并暂存
git rm --cached filename.txt # 仅从暂存区移除（保留本地文件）
```

---

## 4. 代码管理

### 4.1 提交代码

```bash
# 提交代码（打开编辑器输入提交信息）
git commit

# 提交并添加信息（常用）
git commit -m "提交信息"

# 添加并提交（简写）
git commit -am "提交信息"   # 仅适用于已跟踪的文件

# 修改最后一次提交
git commit --amend -m "新的提交信息"
```

### 4.2 查看历史

```bash
# 查看提交历史
git log

# 简洁格式
git log --oneline

# 图形化显示
git log --graph --oneline --all

# 查看指定文件历史
git log filename.txt

# 查看某次提交的内容
git show commit_id
```

### 4.3 撤销操作

```bash
# 撤销工作区修改
git checkout -- filename.txt
git restore filename.txt    # 新版写法

# 撤销暂存区操作
git reset HEAD filename.txt
git restore --staged filename.txt  # 新版写法

# 撤销提交（保留修改）
git reset --soft HEAD~1

# 撤销提交（不保留修改）
git reset --hard HEAD~1

# 恢复某个文件到特定版本
git checkout commit_id -- filename.txt
```

### 4.4 储藏（Stash）

```bash
# 储藏当前修改
git stash
git stash push -m "工作描述"

# 查看储藏列表
git stash list

# 恢复储藏
git stash apply              # 保留储藏
git stash pop                # 恢复并删除储藏

# 删除储藏
git stash drop stash@{0}
git stash clear
```

---

## 5. 分支操作

### 5.1 基本分支操作

```bash
# 查看分支
git branch                   # 本地分支
git branch -r                # 远程分支
git branch -a                # 所有分支

# 创建分支
git branch feature-branch

# 切换分支
git checkout feature-branch
git switch feature-branch    # 新版写法

# 创建并切换
git checkout -b new-branch
git switch -c new-branch     # 新版写法

# 删除分支
git branch -d branch-name    # 已合并
git branch -D branch-name    # 强制删除
```

### 5.2 合并与变基

```bash
# 合并分支
git merge feature-branch

# 变基（使提交历史更整洁）
git rebase main

# 解决冲突后继续变基
git rebase --continue

# 取消变基
git rebase --abort
```

### 5.3 远程分支操作

```bash
# 推送分支到远程
git push origin branch-name
git push -u origin branch-name  # 首次推送并设置上游

# 拉取远程分支
git fetch origin
git pull                       # fetch + merge

# 删除远程分支
git push origin --delete branch-name
```

---

## 6. 协作开发

### 6.1 常用工作流

```
┌─────────────────────────────────────────────────────────┐
│                    Git Flow 工作流                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│    main ─────●─────────●─────────●───────●─────────    │
│              │         │         │        │             │
│              │     发布/Tag   发布/Tag   发布/Tag       │
│              │                                         │
│    develop ──┼────●────●────●────●────●────●────●──     │
│              │    │    │    │    │    │    │            │
│              │    │    └────┘    │    │    │            │
│              │    │              │    │    │            │
│    feature ──┴────┴────●─────────┴────┴────┘            │
│              (功能开发)                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Fork工作流

```bash
# 1. Fork仓库到自己的账户
# 2. 克隆自己的Fork
git clone https://github.com/your-username/repository.git

# 3. 添加上游仓库
git remote add upstream https://github.com/original-owner/repository.git

# 4. 同步最新代码
git fetch upstream
git checkout main
git merge upstream/main

# 5. 创建功能分支开发
git checkout -b feature/new-feature

# 6. 推送并创建Pull Request
git push origin feature/new-feature
```

### 6.3 Pull Request流程

```
1. Fork仓库（可选）
2. 创建功能分支
3. 开发并提交代码
4. 推送到远程
5. 在GitHub创建Pull Request
6. 等待代码审查
7. 根据反馈修改（如需要）
8. 合并代码
9. 清理分支
```

---

## 7. GitHub功能

### 7.1 Issues（问题追踪）

```markdown
# 创建Issue标题和描述

## 描述
详细说明问题或功能请求

## 复现步骤
1. 步骤1
2. 步骤2

## 预期行为
描述期望的结果

## 实际行为
描述实际发生的情况
```

### 7.2 Projects（项目管理）

- 创建看板管理任务
- 使用Kanban、Roadmap等视图
- 关联Issue和PR

### 7.3 Actions（自动化）

```yaml
# .github/workflows/ci.yml 示例
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest
```

### 7.4 Wiki（文档）

- 项目技术文档
- API说明
- 开发指南

### 7.5 Releases（版本发布）

- 创建语义化版本
- 附加发布说明
- 打包下载

---

## 8. 常见问题

### 8.1 解决冲突

```bash
# 1. 先拉取最新代码
git fetch origin
git merge origin/main

# 2. 手动解决冲突文件
# 3. 标记为已解决
git add filename.py

# 4. 完成合并
git commit
```

### 8.2 修改历史提交

```bash
# 修改最近n次提交
git rebase -i HEAD~n

# 命令说明：
# pick - 保留提交
# reword - 修改提交信息
# edit - 暂停修改
# squash - 合并到前一个提交
# drop - 删除提交
```

### 8.3 清理不需要的文件

```bash
# 创建 .gitignore 文件
# 示例内容：
# __pycache__/
# *.pyc
# .env
# node_modules/
# dist/

# 清理缓存
git rm -r --cached .

# 重新添加
git add .
```

### 8.4 常用别名

```bash
# 配置别名
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.df diff
git config --global alias.lg "log --graph --oneline --all"

# 使用别名
git st
git co main
git lg
```

---

## 快速参考表

| 操作 | 命令 |
|------|------|
| 初始化仓库 | `git init` |
| 克隆仓库 | `git clone url` |
| 查看状态 | `git status` |
| 添加修改 | `git add .` |
| 提交代码 | `git commit -m "信息"` |
| 推送代码 | `git push` |
| 拉取代码 | `git pull` |
| 创建分支 | `git checkout -b name` |
| 切换分支 | `git checkout name` |
| 合并分支 | `git merge name` |
| 查看日志 | `git log --oneline` |

---

## 推荐资源

- [Git官方文档](https://git-scm.com/doc)
- [GitHub文档](https://docs.github.com)
- [GitHub Skills](https://skills.github.com)
- [Pro Git电子版](https://git-scm.com/book/zh/v2)

---

*本指南最后更新于 2026年4月*