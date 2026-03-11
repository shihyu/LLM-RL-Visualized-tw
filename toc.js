// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="index.html"><strong aria-hidden="true">1.</strong> 首頁</a></li><li class="chapter-item expanded "><a href="part-01-overview.html"><strong aria-hidden="true">2.</strong> 第1部分：大型模型、強化學習的技術全景圖</a></li><li class="chapter-item expanded "><a href="part-02-llm-basics.html"><strong aria-hidden="true">3.</strong> 第2部分：大型模型基礎</a></li><li class="chapter-item expanded "><a href="part-03-sft.html"><strong aria-hidden="true">4.</strong> 第3部分：SFT（有監督微調）</a></li><li class="chapter-item expanded "><a href="part-04-dpo.html"><strong aria-hidden="true">5.</strong> 第4部分：DPO（直接偏好最佳化）</a></li><li class="chapter-item expanded "><a href="part-05-optimization-without-training.html"><strong aria-hidden="true">6.</strong> 第5部分：免訓練的大型模型最佳化技術</a></li><li class="chapter-item expanded "><a href="part-06-rl-basics.html"><strong aria-hidden="true">7.</strong> 第6部分：強化學習（RL）基礎</a></li><li class="chapter-item expanded "><a href="part-07-policy-optimization.html"><strong aria-hidden="true">8.</strong> 第7部分：策略最佳化架構及衍生演算法</a></li><li class="chapter-item expanded "><a href="part-08-rlhf-rlaif.html"><strong aria-hidden="true">9.</strong> 第8部分：RLHF 與 RLAIF</a></li><li class="chapter-item expanded "><a href="part-09-reasoning.html"><strong aria-hidden="true">10.</strong> 第9部分：邏輯推理（Reasoning）能力最佳化</a></li><li class="chapter-item expanded "><a href="part-10-llm-advanced.html"><strong aria-hidden="true">11.</strong> 第10部分：大型模型基礎拓展</a></li><li class="chapter-item expanded "><a href="appendix.html"><strong aria-hidden="true">12.</strong> 附錄與引用</a></li><li class="chapter-item expanded "><a href="references.html"><strong aria-hidden="true">13.</strong> 參考文獻</a></li><li class="chapter-item expanded "><a href="discussion.html"><strong aria-hidden="true">14.</strong> 交流討論</a></li><li class="chapter-item expanded "><a href="contributing.html"><strong aria-hidden="true">15.</strong> 歡迎建議</a></li><li class="chapter-item expanded "><a href="terms.html"><strong aria-hidden="true">16.</strong> 使用條款</a></li><li class="chapter-item expanded "><a href="citation.html"><strong aria-hidden="true">17.</strong> 引用格式</a></li><li class="chapter-item expanded "><a href="star-history.html"><strong aria-hidden="true">18.</strong> Star 增長趨勢</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
