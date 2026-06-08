import { useSelectParserList } from '@/hooks/use-user-setting-request';
import { useCallback, useMemo } from 'react';

const ParserListMap = new Map([
  [
    ['pdf'],
    [
      'naive',
      'custom',
      'custom_paragraph',
      'resume',
      'manual',
      'paper',
      'book',
      'laws',
      'presentation',
      'one',
      'qa',
      'knowledge_graph',
    ],
  ],
  [
    ['doc', 'docx'],
    [
      'naive',
      'custom',
      'custom_paragraph',
      'resume',
      'book',
      'laws',
      'one',
      'qa',
      'manual',
      'knowledge_graph',
    ],
  ],
  [
    ['xlsx', 'xls'],
    [
      'naive',
      'custom',
      'custom_paragraph',
      'qa',
      'table',
      'one',
      'knowledge_graph',
    ],
  ],
  [['ppt', 'pptx'], ['presentation']],
  [
    ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'webp', 'svg', 'ico'],
    ['picture'],
  ],
  [
    ['txt'],
    [
      'naive',
      'custom',
      'custom_paragraph',
      'resume',
      'book',
      'laws',
      'one',
      'qa',
      'table',
      'knowledge_graph',
    ],
  ],
  [
    ['csv'],
    [
      'naive',
      'custom',
      'custom_paragraph',
      'resume',
      'book',
      'laws',
      'one',
      'qa',
      'table',
      'knowledge_graph',
    ],
  ],
  [
    ['md', 'mdx'],
    ['naive', 'custom', 'custom_paragraph', 'qa', 'knowledge_graph'],
  ],
  [['json'], ['naive', 'custom', 'custom_paragraph', 'knowledge_graph']],
  [['eml'], ['email']],
]);

const getParserList = (
  values: string[],
  parserList: Array<{
    value: string;
    label: string;
  }>,
) => {
  return parserList.filter((x) => values?.some((y) => y === x.value));
};

export const useFetchParserListOnMount = (documentExtension: string) => {
  const parserList = useSelectParserList();

  const nextParserList = useMemo(() => {
    const key = [...ParserListMap.keys()].find((x) =>
      x.some((y) => y === documentExtension),
    );
    if (key) {
      const values = ParserListMap.get(key);
      return getParserList(values ?? [], parserList);
    }

    return getParserList(
      [
        'naive',
        'custom',
        'custom_paragraph',
        'resume',
        'book',
        'laws',
        'one',
        'qa',
        'table',
      ],
      parserList,
    );
  }, [parserList, documentExtension]);

  return { parserList: nextParserList };
};

const hideAutoKeywords = ['qa', 'table', 'resume', 'knowledge_graph', 'tag'];

export const useShowAutoKeywords = () => {
  const showAutoKeywords = useCallback((selectedTag: string) => {
    return hideAutoKeywords.every((x) => selectedTag !== x);
  }, []);

  return showAutoKeywords;
};
