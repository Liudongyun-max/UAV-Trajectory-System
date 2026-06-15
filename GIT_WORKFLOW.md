# UAV Trajectory System 版本管理与 Git 开发规范

为了规范后续的项目管理，我们建议后续所有的功能开发、Bug 修复均采用**分支开发**，并通过 **Pull Request (PR)** 的形式合并到 `development` 分支。只有在需要发布正式稳定版本时，才将 `development` 分支合并到主分支 `main`。同时，使用 **Git Tags** 标记版本，以确保能够随时拉取和定位到历史任意版本。

---

## 🚀 一、日常开发与 PR 提交流程

每次进行新的任务（如：优化轨迹预测算法、打包模型、写测试脚本等）时，请按照以下流程操作：

### 1. 本地创建功能分支
每次开发前，先切换到 `development` 分支拉取最新代码，并创建一个专属的功能分支：
```bash
# 切换到开发分支并拉取最新
git checkout development
git pull origin development

# 创建并切换到新分支（命名推荐：feature/功能名 或 bugfix/修复名）
git checkout -b feature/optimize-gru-model
```

### 2. 本地开发与提交
在新的分支上进行代码开发。开发并测试通过后，提交到本地：
```bash
git add .
git commit -m "feat: optimize GRU model configuration and increase dataset size"
```

### 3. 推送分支到 GitHub
将该功能分支推送到远程仓库：
```bash
git push -u origin feature/optimize-gru-model
```

### 4. 提交 Pull Request (PR) 并合并
1. 打开您的 GitHub 仓库页面，会看到一条黄色的提示栏，提示有新推送的分支，点击 **Compare & pull request**。
2. 页面会自动加载 PR 模板，请根据模板填写**改动内容**和**测试验证结果**。
3. 确认无误后点击 **Create pull request**。
4. 检查 PR 的自动冲突检测，如果没有冲突，点击 **Merge pull request**，随后点击 **Confirm merge**。
5. （可选）合入后，在 GitHub 页面上点击 **Delete branch** 删除已合并的远程分支。

---

## 🏷️ 二、版本管理与标签 (Tag) 发布

当项目开发到某个关键节点，需要发布稳定版本（例如：YOLOv5 在 RK3588 部署完成），我们可以通过**打标签 (Tag)** 的方式将当前代码锁定为特定的版本号。

### 1. 在本地打版本标签并推送
```bash
# 1. 确保当前在 main 分支，且代码是最新的
git checkout main
git pull origin main

# 2. 对当前最新提交打标签（推荐使用语义化版本号，如 v1.0.0）
git tag -a v1.0.0 -m "Release v1.0.0: YOLOv5 and GRU basic prediction system integrated"

# 3. 将标签推送到 GitHub
git push origin v1.0.0
```

### 2. 在 GitHub 网页端创建 Release (推荐)
1. 打开 GitHub 仓库页面，在右侧栏找到 **Releases** 并点击 **Create a new release**。
2. 在 **Choose a tag** 中输入版本号（如 `v1.0.0`），点击创建新标签。
3. 填写 Release 的标题和更新日志，最后点击 **Publish release**。
4. 这样该版本的源代码会被自动打包成 `.zip` 或 `.tar.gz` 供以后随时下载。

---

## 📥 三、如何拉取和切换到指定的历史版本

如果您在后续开发中需要临时切换或拉取之前的某个特定版本：

### 1. 获取最新的标签信息
在本地执行：
```bash
git fetch --tags
```

### 2. 查看所有版本标签
```bash
git tag
# 输出示例：
# v1.0.0
# v1.1.0
```

### 3. 切换到指定的版本标签
如果您想查看或运行某个版本（如 `v1.0.0`）：
```bash
# 切换到对应的 tag。注意：这会使本地处于 "detached HEAD"（游离指针）状态
git checkout v1.0.0
```

### 4. 基于该历史版本创建新分支进行开发
如果您需要在旧版本的基础上做修复或二次开发，请一定要基于该 tag 创建新分支：
```bash
git checkout -b hotfix/fix-old-bug v1.0.0
```
