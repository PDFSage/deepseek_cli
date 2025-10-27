require "language/python"

class DeepseekCli < Formula
  include Language::Python::Virtualenv

  desc "Developer-focused DeepSeek command line tools"
  homepage "https://github.com/GithubM4Max/deepseek_cli"
  head "https://github.com/GithubM4Max/deepseek_cli.git", branch: "main"

  depends_on "python@3.11"

  def install
    virtualenv_install_with_resources
  end

  test do
    out = shell_output("#{bin}/deepseek --version")
    assert_match "0.1.0", out
  end
end
